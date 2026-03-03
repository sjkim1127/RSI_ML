use crate::TensorError;

/// Computes the resulting shape when broadcasting two shapes according to standard rules.
///
/// Standard broadcasting rules:
/// 1. If the shapes do not have the same rank, prepend 1s to the smaller shape until they do.
/// 2. Two dimensions are compatible if they are equal, or if one of them is 1.
/// 3. If compatible, the resulting dimension size is the maximum of the two sizes.
pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>, TensorError> {
    let rank1 = shape1.len();
    let rank2 = shape2.len();
    let out_rank = std::cmp::max(rank1, rank2);

    let mut out_shape = Vec::with_capacity(out_rank);

    for i in 0..out_rank {
        // Compute indices from the back (right-aligned)
        let dim1 = if i < out_rank - rank1 {
            1
        } else {
            shape1[i - (out_rank - rank1)]
        };

        let dim2 = if i < out_rank - rank2 {
            1
        } else {
            shape2[i - (out_rank - rank2)]
        };

        if dim1 == dim2 {
            out_shape.push(dim1);
        } else if dim1 == 1 {
            out_shape.push(dim2);
        } else if dim2 == 1 {
            out_shape.push(dim1);
        } else {
            return Err(TensorError::ShapeMismatch {
                lhs: shape1.to_vec(),
                rhs: shape2.to_vec(),
                op: "broadcast",
            });
        }
    }

    Ok(out_shape)
}

/// Computes strides for a given shape.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0; shape.len()];
    strides[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Maps a linear index from the broadcasted shape back to the original shape's linear index.
pub fn map_broadcast_index(
    broadcasted_linear_idx: usize,
    broadcasted_shape: &[usize],
    broadcasted_strides: &[usize],
    original_shape: &[usize],
    original_strides: &[usize],
) -> usize {
    let mut orig_linear_idx = 0;
    let rank_diff = broadcasted_shape.len() - original_shape.len();

    let mut remaining = broadcasted_linear_idx;
    
    // We iterate over the dimensions of the broadcasted shape.
    for i in 0..broadcasted_shape.len() {
        let dim_idx = remaining / broadcasted_strides[i];
        remaining %= broadcasted_strides[i];

        if i >= rank_diff { // We are within the original shape's dimensions
            let orig_i = i - rank_diff;
            if original_shape[orig_i] > 1 {
                orig_linear_idx += dim_idx * original_strides[orig_i];
            }
            // If original_shape[orig_i] == 1, it was broadcasted, so its contribution to the original linear index is 0.
        }
    }

    orig_linear_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape_same() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_shape_scalar() {
        assert_eq!(broadcast_shape(&[2, 3], &[1]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[1], &[2, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[2, 3], &[]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_shape_1d() {
        assert_eq!(broadcast_shape(&[2, 3], &[3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[2, 3, 4], &[4]).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn test_broadcast_shape_matrix() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 1]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[2, 1], &[1, 3]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_shape_mismatch() {
        assert!(broadcast_shape(&[2, 3], &[2, 4]).is_err());
        assert!(broadcast_shape(&[2, 3], &[4, 3]).is_err());
    }
}
