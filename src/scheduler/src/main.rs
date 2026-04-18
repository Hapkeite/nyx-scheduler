use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, Notify};
use tokio::time::{sleep, Duration};
use std::fs;

mod protocol;
use protocol::{InterceptRequest, SchedulerResponse};

const SOCK_PATH: &str = "/tmp/nyx.sock";
const TELEMETRY_SOCK_PATH: &str = "/tmp/nyx_telemetry.sock";

#[derive(Debug)]
enum JobClass {
    Lightweight,
    ComputeBound,
    MemoryBound,
}

// 1. The Profiler: Analyzes the computational graph's shape
fn classify_job(m: u64, n: u64, k: u64, requested_bytes: u64) -> JobClass {
    // Estimated floating point operations for a matrix multiplication
    let estimated_flops = 2 * m * n * k;
    
    // How math-heavy is this job relative to its memory footprint?
    let compute_density = estimated_flops as f64 / (requested_bytes as f64 + 1.0);

    // Thresholds: Can be tweaked based on your specific GPUs
    if estimated_flops < 500_000_000 && requested_bytes < 100_000_000 {
        JobClass::Lightweight
    } else if compute_density > 50.0 {
        JobClass::ComputeBound
    } else {
        JobClass::MemoryBound
    }
}

// 1. Internal State Structure
#[derive(Debug, Default)]
struct SchedulerState {
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub vram_reserved_bytes: u64, 
    pub gpu_util_pct: u32, 
    pub state_changed: Arc<Notify>, // NEW: Event-driven trigger
}

// Struct matching the JSON emitted by the telemetry daemon
#[derive(Debug, Deserialize)]
struct GpuState {
    gpu_util_pct: u32,
    vram_used_mb: u64,
    vram_total_mb: u64,
    temp_c: u32,
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // Wrap our state in an Arc<Mutex<>> so ownership can be safely shared 
    // across the telemetry listener and all client request tasks.
    let state = Arc::new(Mutex::new(SchedulerState::default()));

    // 2. Spawn the Telemetry Subscriber Task
    let state_clone = Arc::clone(&state);
    tokio::spawn(async move {
        subscribe_to_telemetry(state_clone).await;
    });

    // Start the Interceptor Listener
    if fs::metadata(SOCK_PATH).is_ok() {
        fs::remove_file(SOCK_PATH)?;
    }
    let listener = UnixListener::bind(SOCK_PATH)?;
    println!("Nyx Scheduler listening on {}...", SOCK_PATH);

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let task_state = Arc::clone(&state);
                // Spawn a task for each PyTorch DL job
                tokio::spawn(async move {
                    if let Err(e) = handle_client(stream, task_state).await {
                        eprintln!("Error handling client: {}", e);
                    }
                });
            }
            Err(e) => eprintln!("Failed to accept connection: {}", e),
        }
    }
}

// Connects to Nitin's NVML stream and updates the internal state continuously
async fn subscribe_to_telemetry(state: Arc<Mutex<SchedulerState>>) {
    loop {
        match UnixStream::connect(TELEMETRY_SOCK_PATH).await {
            Ok(stream) => {
                println!("Connected to Telemetry Daemon.");
                let mut reader = BufReader::new(stream);
                let mut line = String::new();

                while let Ok(bytes_read) = reader.read_line(&mut line).await {
                    if bytes_read == 0 { break; } // Stream closed

                    if let Ok(gpu_state) = serde_json::from_str::<GpuState>(&line) {
                        let mut s = state.lock().await;
                        s.vram_total_mb = gpu_state.vram_total_mb;
                        s.vram_used_mb = gpu_state.vram_used_mb;
                        s.gpu_util_pct = gpu_state.gpu_util_pct;
                        s.vram_reserved_bytes = 0; 
                        
                        // Wake up any tasks waiting for VRAM or Compute to free up!
                        s.state_changed.notify_waiters();
                    }
                    line.clear();
                }
            }
            Err(_) => {
                // If telemetry isn't up yet, wait and retry
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
}

// 3. The Hold/Release Queuing Logic
async fn handle_client(mut stream: UnixStream, state: Arc<Mutex<SchedulerState>>) -> std::io::Result<()> {
    let mut buffer = [0; 1024];

    loop {
        let n = stream.read(&mut buffer).await?;
        if n == 0 { break; }

        let msg_str = String::from_utf8_lossy(&buffer[..n]);
        
        match serde_json::from_str::<InterceptRequest>(&msg_str) {
            Ok(InterceptRequest::Malloc { bytes }) => {
                println!("Client requesting {} bytes...", bytes);
                
                // Grab a clone of the Arc pointing to the Notify flag
                let notify = state.lock().await.state_changed.clone();
                
                loop {
                    // Create the notification future BEFORE locking to prevent race conditions
                    let notified = notify.notified();
                    
                    {
                        let mut s = state.lock().await;
                        let vram_used_bytes = (s.vram_used_mb * 1024 * 1024) + s.vram_reserved_bytes;
                        let vram_total_bytes = s.vram_total_mb * 1024 * 1024;
                        
                        // Safety margin: ensure we have the bytes available
                        if vram_used_bytes + (bytes as u64) < vram_total_bytes {
                            // Reserve the memory
                            s.vram_reserved_bytes += bytes as u64;
                            println!("Granted {} bytes. Total reserved: {}", bytes, s.vram_reserved_bytes);
                            break; 
                        }
                    } // Explicitly drop lock before awaiting
                    
                    // Wait instantly for an event instead of an arbitrary sleep
                    notified.await;
                }

                let response = SchedulerResponse { status: "Go".to_string() };
                stream.write_all(serde_json::to_string(&response).unwrap().as_bytes()).await?;
            }
            Ok(InterceptRequest::Free { ptr }) => {
                println!("Client freeing ptr: {}", ptr);
                
                // Notify waiters that memory might have freed up!
                state.lock().await.state_changed.notify_waiters();
                
                let response = SchedulerResponse { status: "Go".to_string() };
                stream.write_all(serde_json::to_string(&response).unwrap().as_bytes()).await?;
            }
            Ok(InterceptRequest::Cublas_sgemm { m, n, k }) => {
                println!("Intercepted Graph Shape: m={}, n={}, k={}", m, n, k);
                
                let estimated_vram = 10_485_760; 
                let job_class = classify_job(m, n, k, estimated_vram);
                
                println!("Job Classified as: {:?}", job_class);

                let notify = state.lock().await.state_changed.clone();

                loop {
                    let notified = notify.notified();
                    
                    {
                        let mut s = state.lock().await;
                        let mut should_grant = false;

                        match job_class {
                            JobClass::Lightweight => {
                                if s.gpu_util_pct < 95 { should_grant = true; }
                            }
                            JobClass::ComputeBound => {
                                if s.gpu_util_pct < 75 { should_grant = true; }
                            }
                            JobClass::MemoryBound => {
                                let vram_used_bytes = (s.vram_used_mb * 1024 * 1024) + s.vram_reserved_bytes;
                                if vram_used_bytes + estimated_vram < (s.vram_total_mb * 1024 * 1024) {
                                    should_grant = true;
                                    s.vram_reserved_bytes += estimated_vram; 
                                }
                            }
                        }

                        if should_grant {
                            break; // Send the "Go" signal to the C++ Interceptor
                        }
                    }

                    notified.await;
                }

                let response = SchedulerResponse { status: "Go".to_string() };
                stream.write_all(serde_json::to_string(&response).unwrap().as_bytes()).await?;
            }
            Ok(InterceptRequest::Compute { grid_x, block_x }) => {
                println!("Compute slice requested (Grid X: {}, Block X: {})", grid_x, block_x);
                
                let notify = state.lock().await.state_changed.clone();

                loop {
                    let notified = notify.notified();
                    
                    {
                        let s = state.lock().await;
                        if s.gpu_util_pct < 90 {
                            break; 
                        }
                    }
                    
                    notified.await;
                }
                
                let response = SchedulerResponse { status: "Go".to_string() };
                stream.write_all(serde_json::to_string(&response).unwrap().as_bytes()).await?;
            }
            Err(e) => eprintln!("Failed to parse message: {}", e),
        }
    }
    Ok(())
}