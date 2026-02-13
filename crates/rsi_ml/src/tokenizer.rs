use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct CharTokenizer {
    stoi: HashMap<char, usize>,
    itos: Vec<char>,
}

impl CharTokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();

        let mut stoi = HashMap::with_capacity(chars.len());
        for (idx, ch) in chars.iter().enumerate() {
            stoi.insert(*ch, idx);
        }

        Self { stoi, itos: chars }
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| self.stoi.get(&c).copied().unwrap_or(0))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|id| self.itos.get(*id).copied().unwrap_or('?'))
            .collect()
    }
}
