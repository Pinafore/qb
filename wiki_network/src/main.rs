extern crate wiki_network;

use std::io::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::collections::{HashSet, HashMap};

use wiki_network::wikipedia::*;

fn main() {
    let stop_words = load_stopwords();
    let questions = load_questions();
    let wiki_graph = WikiGraph::new();
    let ref new_page_map = wiki_graph.new_page_map;
    let ref neighbors = wiki_graph.adjacency_list;
    let ref vertex_page_map = wiki_graph.vertex_page_map;
    println!("Starting question guessing");
    for (i, q) in questions.iter().enumerate() {
        println!("Starting iteration {}", i);
        let tokens = tokenize(q, &stop_words);
        let mut seeds: HashSet<usize> = HashSet::new();
        let mut candidates: HashSet<usize> = HashSet::new();
        for word in tokens {
            if new_page_map.contains_key(&word) {
                let v_index = new_page_map[&word];
                seeds.insert(v_index);
                for node in &neighbors[v_index] {
                    candidates.insert(*node);
                }
            }
        }
        let sources: Vec<usize> = seeds.into_iter().collect();
        let destinations: Vec<usize> = candidates.into_iter().collect();
        let distances = path_lengths(&wiki_graph, &sources, &destinations, 6);
        let mut candidate_distances: HashMap<usize, usize> = HashMap::new();
        for (seed, candidate, dist) in distances {
            if !candidate_distances.contains_key(&candidate) {
                candidate_distances.insert(candidate, 1 / dist);
            } else {
                let new_dist = candidate_distances[&candidate] + dist;
                candidate_distances.insert(candidate, new_dist);
            }
        }
        let mut candidate_dist_vec: Vec<(usize, usize)> = candidate_distances.into_iter().collect();
        candidate_dist_vec.sort_by_key(|entry| entry.1);
        for (candidate, dist) in candidate_dist_vec {
            let ref page = vertex_page_map[&candidate];
            println!("{}:{}", page, dist);
        }

    }
    let albert = "albert_einstein";
    let v_albert = new_page_map[albert];
    println!("Remaps found");
    let ref albert_neighbors = neighbors[v_albert];
    println!("Num: {}", albert_neighbors.len());
    for n in albert_neighbors {
        println!("Neighbor: {}", vertex_page_map[n]);
    }
}

