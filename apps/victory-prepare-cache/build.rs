use std::{env, fs, path::PathBuf};

fn main() {
    let source = PathBuf::from("../../crates/victory-gpu/src/shader_cache.rs");
    println!("cargo:rerun-if-changed={}", source.display());
    let code = fs::read_to_string(source).expect("renderer shader cache source");
    let original = "std::env::current_exe().and_then(std::fs::read)";
    assert_eq!(code.matches(original).count(), 1);
    let mut code = code.replace(original, "std::fs::read(std::env::args_os().nth(2).expect(\"renderer executable\"))");
    code.push_str("\npub fn compiled(input: &[u8]) -> Result<Vec<u8>, String> { host_shader(input).map(|s| encode(&s)) }\npub fn stats() -> (u64, u64) { counts() }\n");
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let module = output.join("host_cache.rs");
    fs::write(&module, code).unwrap();
    fs::write(output.join("include.rs"), format!("#[allow(dead_code)] #[path = {module:?}] mod host_cache;")).unwrap();
}
