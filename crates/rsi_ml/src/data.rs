#[derive(Clone, Debug)]
pub struct SequenceDataset {
    pub tokens: Vec<usize>,
    pub seq_len: usize,
}

impl SequenceDataset {
    pub fn new(tokens: Vec<usize>, seq_len: usize) -> Self {
        Self { tokens, seq_len }
    }

    pub fn len(&self) -> usize {
        self.tokens.len().saturating_sub(self.seq_len + 1)
    }

    pub fn sample_pair(&self, start: usize) -> (&[usize], &[usize]) {
        let x_start = start;
        let x_end = start + self.seq_len;
        let y_start = start + 1;
        let y_end = y_start + self.seq_len;
        (&self.tokens[x_start..x_end], &self.tokens[y_start..y_end])
    }
}

#[derive(Clone, Debug)]
pub struct SequenceBatchIter<'a> {
    dataset: &'a SequenceDataset,
    batch_size: usize,
    cursor: usize,
}

impl<'a> SequenceBatchIter<'a> {
    pub fn new(dataset: &'a SequenceDataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            cursor: 0,
        }
    }

    pub fn next_batch_indices(&mut self) -> Vec<usize> {
        let dataset_len = self.dataset.len();
        if dataset_len == 0 {
            return vec![];
        }
        let mut out = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            out.push(self.cursor % dataset_len);
            self.cursor = (self.cursor + 1) % dataset_len;
        }
        out
    }
}

pub fn one_hot(ids: &[usize], vocab_size: usize) -> Vec<f32> {
    let mut out = vec![0.0; ids.len() * vocab_size];
    for (row, id) in ids.iter().enumerate() {
        out[row * vocab_size + *id] = 1.0;
    }
    out
}
