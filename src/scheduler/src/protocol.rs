use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(tag = "action", rename_all = "lowercase")]
pub enum InterceptRequest {
    Malloc { bytes: usize },
    Free { ptr: String },
    Compute { grid_x: u32, block_x: u32 }, 
}

#[derive(Debug, Serialize)]
pub struct SchedulerResponse {
    pub status: String,
}