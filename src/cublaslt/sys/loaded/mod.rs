#[cfg(feature = "cuda-12080")]
mod sys_12080;
#[cfg(feature = "cuda-12080")]
pub use sys_12080::*;

pub unsafe fn culib() -> &'static Lib {
    static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
    LIB.get_or_init(|| {
        let lib_names = ["cublasLt"];
        let choices: Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = Lib::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}

mod adapter;
pub use adapter::*;
