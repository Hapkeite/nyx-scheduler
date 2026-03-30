use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use std::fs;

mod protocol;
use protocol::{InterceptRequest, SchedulerResponse};

const SOCK_PATH: &str = "/tmp/nyx.sock";
const TELEMETRY_SOCK_PATH: &str = "/tmp/nyx_telemetry.sock";

// 1. Internal State Structure
#[derive(Debug, Default)]
struct SchedulerState {
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    // We track "reserved" bytes to prevent overcommitting VRAM 
    // before the telemetry daemon catches up to the new allocations.
    pub vram_reserved_bytes: u64, 
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
                        // Reset reserved bytes periodically as telemetry catches up
                        s.vram_reserved_bytes = 0; 
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
                
                // Queuing logic: Loop and yield until sufficient VRAM is available
                loop {
                    let mut s = state.lock().await;
                    let vram_used_bytes = (s.vram_used_mb * 1024 * 1024) + s.vram_reserved_bytes;
                    let vram_total_bytes = s.vram_total_mb * 1024 * 1024;
                    
                    // Safety margin: ensure we have the bytes available
                    if vram_used_bytes + (bytes as u64) < vram_total_bytes {
                        // We have space! Reserve it internally so we don't double-allocate 
                        // before the next telemetry tick.
                        s.vram_reserved_bytes += bytes as u64;
                        println!("Granted {} bytes. Total reserved: {}", bytes, s.vram_reserved_bytes);
                        break; 
                    }
                    
                    // Explicitly drop the Mutex lock so other tasks can update the state while we wait
                    drop(s);
                    
                    // "Wait" by sleeping the task, holding the CUDA call intercepted
                    sleep(Duration::from_millis(50)).await;
                }

                // Release the CUDA call
                let response = SchedulerResponse { status: "Go".to_string() };
                stream.write_all(serde_json::to_string(&response).unwrap().as_bytes()).await?;
            }
            Ok(InterceptRequest::Free { ptr }) => {
                println!("Client freeing ptr: {}", ptr);
                // We let the NVML telemetry physically catch the drop, but we acknowledge it
                let response = SchedulerResponse { status: "Go".to_string() };
                stream.write_all(serde_json::to_string(&response).unwrap().as_bytes()).await?;
            }
            // NEW: Handle compute requests
            Ok(InterceptRequest::Compute { grid_x, block_x }) => {
                println!("Client requesting compute slice! (Grid X: {}, Block X: {})", grid_x, block_x);
                
                // TODO: Future Time-Slicing Logic. 
                // If another client is actively computing, we would sleep() here!
                
                let response = SchedulerResponse { status: "Go".to_string() };
                stream.write_all(serde_json::to_string(&response).unwrap().as_bytes()).await?;
            }
            Err(e) => eprintln!("Failed to parse message: {}", e),
        }
    }
    Ok(())
}