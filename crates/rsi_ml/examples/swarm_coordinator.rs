use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Debug)]
struct WorkerReport {
    worker: String,
    round: usize,
    best_seed: u64,
    base_score: f32,
    future_score: f32,
    effective_score: f32,
}

fn parse_report(worker: String, content: &str) -> Option<WorkerReport> {
    let line = content.lines().next()?.trim();
    let mut parts = line.split(',');
    let round = parts.next()?.parse::<usize>().ok()?;
    let best_seed = parts.next()?.parse::<u64>().ok()?;
    let p3 = parts.next()?.parse::<f32>().ok()?;
    let p4 = parts.next().and_then(|v| v.parse::<f32>().ok());
    let p5 = parts.next().and_then(|v| v.parse::<f32>().ok());
    let (base_score, future_score, effective_score) = if let (Some(p4), Some(p5)) = (p4, p5) {
        (p3, p4, p5)
    } else {
        // Backward compatibility with old "round,seed,score" format.
        (p3, p3, p3)
    };
    Some(WorkerReport {
        worker,
        round,
        best_seed,
        base_score,
        future_score,
        effective_score,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let out_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "swarm_out".to_string());

    let mut reports = Vec::new();
    for entry in fs::read_dir(&out_dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(v) => v.to_string(),
            None => continue,
        };
        if !name.starts_with("worker_") || !name.ends_with("_latest.txt") {
            continue;
        }

        let worker = name
            .trim_start_matches("worker_")
            .trim_end_matches("_latest.txt")
            .to_string();
        let content = fs::read_to_string(&path)?;
        if let Some(r) = parse_report(worker, &content) {
            reports.push(r);
        }
    }

    if reports.is_empty() {
        println!("no worker reports found in {}", out_dir);
        return Ok(());
    }

    reports.sort_by(|a, b| {
        a.effective_score
            .partial_cmp(&b.effective_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("swarm leaderboard (lower effective is better):");
    for r in &reports {
        println!(
            "worker={} round={} seed={} base={:.6} future={:.6} effective={:.6}",
            r.worker, r.round, r.best_seed, r.base_score, r.future_score, r.effective_score
        );
    }

    let best = &reports[0];
    let best_path = PathBuf::from(&out_dir).join("global_best_seed.txt");
    fs::write(
        &best_path,
        format!(
            "worker={},round={},seed={},base_score={},future_score={},effective_score={}\n",
            best.worker,
            best.round,
            best.best_seed,
            best.base_score,
            best.future_score,
            best.effective_score
        ),
    )?;

    let leaderboard_path = PathBuf::from(&out_dir).join("leaderboard.csv");
    let mut csv = String::from("worker,round,best_seed,base_score,future_score,effective_score\n");
    for r in &reports {
        csv.push_str(&format!(
            "{},{},{},{},{},{}\n",
            r.worker, r.round, r.best_seed, r.base_score, r.future_score, r.effective_score
        ));
    }
    fs::write(&leaderboard_path, csv)?;

    println!(
        "global best: worker={} seed={} effective={:.6} saved={}",
        best.worker,
        best.best_seed,
        best.effective_score,
        best_path.display()
    );
    Ok(())
}
