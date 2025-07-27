use hf_hub::api::sync::Api;
use std::env;
use std::fs;
use std::path::Path;

fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let model_dir = Path::new(&out_dir).join("model");

    // Create model directory if it doesn't exist
    fs::create_dir_all(&model_dir)?;

    // Check if model files already exist
    let tokenizer_path = model_dir.join("tokenizer.json");
    let model_path = model_dir.join("model.safetensors");
    let config_path = model_dir.join("config.json");

    if !tokenizer_path.exists() || !model_path.exists() || !config_path.exists() {
        println!("cargo:info=Downloading minishlab/potion-base-8M model...");

        // Initialize HF Hub API
        let api = Api::new().expect("hf-hub API init failed");
        let repo = api.model("minishlab/potion-base-8M".to_string());

        // Download model files
        let tokenizer_file = repo
            .get("tokenizer.json")
            .expect("Failed to download tokenizer.json");
        let model_file = repo
            .get("model.safetensors")
            .expect("Failed to download model.safetensors");
        let config_file = repo
            .get("config.json")
            .expect("Failed to download config.json");

        // Copy files to output directory
        fs::copy(&tokenizer_file, &tokenizer_path)?;
        fs::copy(&model_file, &model_path)?;
        fs::copy(&config_file, &config_path)?;

        println!("cargo:info=Model download completed successfully");
    }

    // Generate constants for the embedded file paths
    let constants_code = format!(
        r#"
pub const TOKENIZER_BYTES: &[u8] = include_bytes!(r"{}");
pub const MODEL_BYTES: &[u8] = include_bytes!(r"{}");
pub const CONFIG_BYTES: &[u8] = include_bytes!(r"{}");
"#,
        tokenizer_path.display(),
        model_path.display(),
        config_path.display()
    );

    let constants_path = Path::new(&out_dir).join("model_constants.rs");
    fs::write(&constants_path, constants_code)?;

    Ok(())
}
