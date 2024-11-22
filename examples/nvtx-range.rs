use cudarc::nvtx::safe::{mark, range_end, range_pop, range_push, range_start};


fn main() {
    range_push("Test Range");
    std::thread::sleep(std::time::Duration::from_secs(1));
    range_pop();

    range_push("Test Range2");
    std::thread::sleep(std::time::Duration::from_secs(1));
    range_push("Test Range3");
    std::thread::sleep(std::time::Duration::from_secs(1));
    range_pop();
    range_pop();

    let id = range_start("Test Range4");
    std::thread::sleep(std::time::Duration::from_secs(1));
    mark("Test Mark");
    std::thread::sleep(std::time::Duration::from_secs(1));
    range_end(id);
}
