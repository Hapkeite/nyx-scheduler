use nvml_wrapper::Nvml;
use serde::Serialize;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::os::unix::net::UnixListener;
use std::thread;
use std::time::Duration;

const SOCKET_PATH: &str = "/tmp/nyx_telemetry.sock";

#[derive(Serialize)]
struct GpuState {
    gpu_util_pct: u32,
    vram_used_mb: u64,
    vram_total_mb: u64,
    temp_c: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Force nvml-wrapper to load the exact driver file using the builder
    let nvml = Nvml::builder()
        .lib_path(OsStr::new("/lib/x86_64-linux-gnu/libnvidia-ml.so.1"))
        .init()?;
    
    let device = nvml.device_by_index(0)?;
    println!("Telemetry Daemon Online. Monitoring: {}", device.name()?);

    if fs::metadata(SOCKET_PATH).is_ok() {
        fs::remove_file(SOCKET_PATH)?;
    }

    let listener = UnixListener::bind(SOCKET_PATH)?;
    println!("Broadcasting telemetry on {}", SOCKET_PATH);

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                println!("Client connected. Starting telemetry stream...");

                thread::spawn(move || {
                    // 2. Also update the thread's Nvml instance
                    let nvml_thread = Nvml::builder()
                        .lib_path(OsStr::new("/lib/x86_64-linux-gnu/libnvidia-ml.so.1"))
                        .init()
                        .unwrap();
                        
                    let device_thread = nvml_thread.device_by_index(0).unwrap();

                    loop {
                        let util = device_thread.utilization_rates().unwrap();
                        let mem = device_thread.memory_info().unwrap();
                        let temp = device_thread
                            .temperature(
                                nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu,
                            )
                            .unwrap();

                        let state = GpuState {
                            gpu_util_pct: util.gpu,
                            vram_used_mb: mem.used / (1024 * 1024),
                            vram_total_mb: mem.total / (1024 * 1024),
                            temp_c: temp,
                        };

                        let mut json_msg = serde_json::to_string(&state).unwrap();
                        json_msg.push('\n');

                        if let Err(e) = stream.write_all(json_msg.as_bytes()) {
                            println!("Client disconnected: {}", e);
                            break;
                        }

                        thread::sleep(Duration::from_millis(100));
                    }
                });
            }
            Err(err) => {
                eprintln!("Connection failed: {}", err);
            }
        }
    }

    Ok(())
}