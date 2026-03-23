use serde::{Deserialize, Serialize};
use std::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::time::{Duration, timeout};

#[derive(Deserialize, Debug)]
struct InterceptorMessage {
    action: String,
    bytes: Option<u64>,
}

#[derive(Serialize, Debug)]
struct DaemonResponse {
    status: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let socket_path = "/tmp/nyx.sock";
    if fs::metadata(socket_path).is_ok() {
        fs::remove_file(socket_path)?;
    }

    let listener = UnixListener::bind(socket_path)?;
    println!(
        "🚀 [ASYNC IPC] Scheduler Brain online! Listening on {}...",
        socket_path
    );

    loop {
        if let Ok((stream, _)) = listener.accept().await {
            tokio::spawn(async move {
                handle_client(stream).await;
            });
        }
    }
}

async fn handle_client(mut stream: UnixStream) {
    let mut buffer = vec![0; 1024];
    if let Ok(Ok(size)) = timeout(Duration::from_secs(2), stream.read(&mut buffer)).await {
        if size == 0 {
            return;
        }

        let raw_msg = String::from_utf8_lossy(&buffer[..size]);
        if let Ok(req) = serde_json::from_str::<InterceptorMessage>(&raw_msg) {
            println!(
                "📥 [REQUEST] Action: {}, Bytes: {:?}",
                req.action, req.bytes
            );

            let res = serde_json::to_string(&DaemonResponse {
                status: "GO".into(),
            })
            .unwrap();
            let _ = stream.write_all(res.as_bytes()).await;
        }
    }
}
