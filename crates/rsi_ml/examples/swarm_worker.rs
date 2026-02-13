use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use rsi_ml::{mse, Genome, Tensor};

fn next_rand(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn parse_arg<T: std::str::FromStr>(args: &[String], idx: usize, default: T) -> T {
    args.get(idx)
        .and_then(|s| s.parse::<T>().ok())
        .unwrap_or(default)
}

fn evaluate_seed(seed: u64, x: &Tensor, y: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
    let genome = Genome::from_seed_mlp(seed, 2, 2, 1)?;
    let pred = genome.forward(x)?;
    let task = mse(&pred, y)?.eval()[0];
    Ok(task + genome.complexity_score() as f32 * 0.0001)
}

fn rand_unit_f32(state: &mut u64) -> f32 {
    let v = next_rand(state);
    (v as f64 / u64::MAX as f64) as f32
}

fn mutate_near_seed(anchor: u64, state: &mut u64, radius_bits: usize) -> u64 {
    let bounded = radius_bits.clamp(1, 64);
    let mut seed = anchor;
    let flips = ((next_rand(state) % 3) + 1) as usize;
    for _ in 0..flips {
        let bit = (next_rand(state) % bounded as u64) as u32;
        seed ^= 1_u64 << bit;
    }
    seed
}

fn parse_global_seed(content: &str) -> Option<u64> {
    for token in content.split([',', '\n', ' ']) {
        let trimmed = token.trim();
        if let Some(v) = trimmed.strip_prefix("seed=") {
            if let Ok(seed) = v.parse::<u64>() {
                return Some(seed);
            }
        } else if let Ok(seed) = trimmed.parse::<u64>() {
            return Some(seed);
        }
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let worker_id = args.get(1).cloned().unwrap_or_else(|| "w0".to_string());
    let rounds: usize = parse_arg(&args, 2, 20usize);
    let candidates_per_round: usize = parse_arg(&args, 3, 64usize);
    let base_seed: u64 = parse_arg(&args, 4, 1234u64);
    let out_dir = args
        .get(5)
        .cloned()
        .unwrap_or_else(|| "swarm_out".to_string());
    let lookahead_steps: usize = parse_arg(&args, 6, 3usize);
    let lookahead_alpha: f32 = parse_arg(&args, 7, 0.5f32);
    let pull_every: usize = parse_arg(&args, 8, 2usize);
    let exploit_ratio: f32 = parse_arg(&args, 9, 0.7f32);
    let local_radius_bits: usize = parse_arg(&args, 10, 12usize);
    let lookahead_enabled = lookahead_steps > 0 && lookahead_alpha != 0.0;

    fs::create_dir_all(&out_dir)?;
    let csv_path = PathBuf::from(&out_dir).join(format!("worker_{}.csv", worker_id));
    let latest_path = PathBuf::from(&out_dir).join(format!("worker_{}_latest.txt", worker_id));
    let global_best_path = PathBuf::from(&out_dir).join("global_best_seed.txt");

    let mut csv = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&csv_path)?;
    if csv.metadata()?.len() == 0 {
        writeln!(csv, "round,best_seed,base_score,future_score,effective_score")?;
    }

    // Base task (present-time fit)
    let x = Tensor::from_loaded(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2], false)?;
    let y = Tensor::from_loaded(vec![3.0, 3.0], vec![2, 1], false)?;
    // Future task (small distribution shift proxy)
    let x_future = Tensor::from_loaded(vec![1.2, 1.8, 2.1, 0.9], vec![2, 2], false)?;
    let y_future = Tensor::from_loaded(vec![3.0, 3.1], vec![2, 1], false)?;

    let mut rng = base_seed ^ (worker_id.bytes().map(|b| b as u64).sum::<u64>() << 1);
    let mut global_best_seed = base_seed;
    let mut global_best_base = f32::INFINITY;
    let mut global_best_future = f32::INFINITY;
    let mut global_best_effective = f32::INFINITY;
    let mut pulled_global = false;

    for round in 0..rounds {
        let mut anchor_seed = global_best_seed;
        if pull_every > 0 && round % pull_every == 0 {
            if let Ok(content) = fs::read_to_string(&global_best_path) {
                if let Some(seed) = parse_global_seed(&content) {
                    anchor_seed = seed;
                    pulled_global = true;
                }
            }
        }

        let mut round_best_seed = global_best_seed;
        let mut round_best_base = global_best_base;
        let mut round_best_future = global_best_future;
        let mut round_best_effective = global_best_effective;
        let mut exploit_count = 0usize;
        let mut explore_count = 0usize;

        for _ in 0..candidates_per_round {
            let p = rand_unit_f32(&mut rng);
            let do_exploit = p < exploit_ratio;
            let candidate_seed = if do_exploit {
                exploit_count += 1;
                mutate_near_seed(anchor_seed, &mut rng, local_radius_bits)
            } else {
                explore_count += 1;
                next_rand(&mut rng)
            };
            let base_score = evaluate_seed(candidate_seed, &x, &y)?;
            let future_score = if lookahead_enabled {
                evaluate_seed(candidate_seed, &x_future, &y_future)?
            } else {
                base_score
            };
            let effective_score = base_score + lookahead_alpha * (future_score - base_score);
            if effective_score < round_best_effective {
                round_best_seed = candidate_seed;
                round_best_base = base_score;
                round_best_future = future_score;
                round_best_effective = effective_score;
            }
        }

        global_best_seed = round_best_seed;
        global_best_base = round_best_base;
        global_best_future = round_best_future;
        global_best_effective = round_best_effective;

        writeln!(
            csv,
            "{},{},{},{},{}",
            round,
            global_best_seed,
            global_best_base,
            global_best_future,
            global_best_effective
        )?;

        fs::write(
            &latest_path,
            format!(
                "{},{},{},{},{}\n",
                round, global_best_seed, global_best_base, global_best_future, global_best_effective
            ),
        )?;

        if round % 5 == 0 {
            println!(
                "worker={} round={} best_seed={} base={:.6} future={:.6} effective={:.6} pulled_global={} exploit={} explore={}",
                worker_id,
                round,
                global_best_seed,
                global_best_base,
                global_best_future,
                global_best_effective,
                pulled_global,
                exploit_count,
                explore_count
            );
        }
    }

    println!(
        "worker={} done rounds={} final_best_seed={} base={:.6} future={:.6} effective={:.6} lookahead_steps={} lookahead_alpha={} pull_every={} exploit_ratio={} local_radius_bits={} out_dir={}",
        worker_id,
        rounds,
        global_best_seed,
        global_best_base,
        global_best_future,
        global_best_effective,
        lookahead_steps,
        lookahead_alpha,
        pull_every,
        exploit_ratio,
        local_radius_bits,
        out_dir
    );
    Ok(())
}
