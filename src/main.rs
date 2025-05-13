#[tokio::main(flavor = "current_thread")]
async fn main() {
    use tokio::sync::mpsc;
    use tokio::task;

    let (tx, mut rx) = mpsc::channel(32);

    // Spawn a producer task to simulate adding tasks to the queue
    task::spawn(async move {
        for i in 1..=10 {
            if tx.send(format!("Task {}", i)).await.is_err() {
                println!("Receiver dropped");
                break;
            }
        }
    });

    // Consumer loop
    while let Some(task) = rx.recv().await {
        println!("Processing {}", task);
    }
}
