use crate::events::Event;

mod events;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    use tokio::sync::mpsc;
    use tokio::task;

    let (tx, mut rx) = mpsc::channel(32);

    // Spawn a producer task to simulate adding tasks to the queue
    task::spawn(async move {
        for i in 1..=10 {
            if tx.send(Event::KeyboardInput(i)).await.is_err() {
                println!("Receiver dropped");
                break;
            }
        }
    });

    // Consumer loop
    while let Some(event) = rx.recv().await {
        match event {
            Event::KeyboardInput(key) => println!("Key event {}", key),
        }
    }
}
