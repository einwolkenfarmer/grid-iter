#![no_std]
//! Grids for Iterators
//!
//! Provides a two dimensional abstraction over Iterators.
//! Intended to be simple, flexible and ideomatic.
//! ```rust
//! use grid_iter::IntoGridIter;
//!
//! let file:&str = "1,2,3,4,5\n6,7,8,9,10\n11,12,13,14,15";
//! let columns = file.find('\n').unwrap();
//! let mut store = file.lines()
//!     .flat_map(|line|line.split(',').map(|s|s.parse().unwrap()))
//!     .collect::<Vec<_>>();
//! store.iter_mut().into_grid_iter(columns).iter_col(3).for_each(|i| *i= 0);
//! store.iter_mut().into_grid_iter(columns).iter_row(1).for_each(|i| *i+= 1);
//! let borrowing_grid = store.iter().into_grid_iter(5);
//! drop(borrowing_grid);
//! let capturing_grid = store.iter().into_grid_iter(5);
//! println!("{:?}", capturing_grid);
//! ```
use core::iter::{Skip, StepBy, Take, repeat};

///The Grid struct wraps an Iterator and provies two dimensional access over its contents.
#[derive(Copy, Debug)]
pub struct GridIter<I: Iterator<Item = T>, T> {
    inner: I,
    columns: usize,
    rows: Option<usize>,
}
//M anually implement because T does not need to be Clone
impl<I: Iterator<Item = T> + Clone, T> Clone for GridIter<I, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            columns: self.columns.clone(),
            rows: self.rows.clone(),
        }
    }
}
impl<I: Iterator<Item = T> + Clone, T: core::fmt::Display> core::fmt::Display for GridIter<I, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.clone().iter_rows().for_each(|col| {
            col.for_each(|ch| write!(f, "{}\t", ch).unwrap());
            write!(f, "\n").unwrap();
        });
        Ok(())
    }
}

/// IntoGridIter ist implemented for all iterators.
/// Provides the grid function to wrap iterators with the Grid struct which contains the main functionality.
pub trait IntoGridIter<I: Iterator<Item = T>, T> {
    fn into_grid_iter(self, columns: usize) -> GridIter<I, T>;
}

impl<I: Iterator<Item = T>, T> IntoGridIter<I, T> for I {
    fn into_grid_iter(self, columns: usize) -> GridIter<I, T> {
        GridIter {
            inner: self,
            columns,
            rows: None,
        }
    }
}

pub type Get<T> = Take<Skip<T>>;
pub type RowIter<T> = Take<Skip<T>>;
pub type ColIter<T> = StepBy<Skip<T>>;
pub type DiagBwdIter<T> = Take<StepBy<Skip<T>>>;
pub type DiagFwdIter<T> = Take<StepBy<Skip<T>>>;

impl<I: Iterator<Item = T>, T> GridIter<I, T> {
    ///```rust
    ///
    /// // . x .
    /// // . x .
    /// // . x .
    ///
    /// use grid_iter::IntoGridIter;
    /// (0..25).into_grid_iter(5)
    ///     .iter_col(3)
    ///     .zip([3,8,13,18,23])
    ///     .for_each(|(l, r)| assert!(l == r));
    ///```   
    pub fn iter_col(self, col: usize) -> ColIter<I> {
        assert!(col < self.columns);
        self.inner.into_iter().skip(col).step_by(self.columns)
    }
    ///```rust
    ///
    /// // . . .
    /// // x x x
    /// // . . .
    ///
    /// use grid_iter::IntoGridIter;
    /// (0..25).into_grid_iter(5)
    ///     .iter_row(3)
    ///     .zip(15..20)
    ///     .for_each(|(l, r)| assert!(l == r));
    ///```
    pub fn iter_row(self, row: usize) -> RowIter<I> {
        self.inner.skip(row * self.columns).take(self.columns)
    }

    ///```rust
    ///
    /// // . . .
    /// // . x .
    /// // . . .
    ///
    /// use grid_iter::IntoGridIter;
    /// assert!((0..25).into_grid_iter(5)
    ///     .get(2,2)==Some(12))
    ///```
    pub fn get(self, col: usize, row: usize) -> Option<<I as IntoIterator>::Item> {
        self.inner
            .skip(index_to_flat(self.columns, col, row))
            .take(1)
            .next()
    }
    pub fn get_kernel(self, col: usize, row: usize) -> Option<[<I as IntoIterator>::Item; 9]> {
        if col == 0 || col == self.columns - 1 || row == 0 {
            None
        } else {
            let mut iter = self
                .inner
                .skip(index_to_flat(self.columns, col - 1, row - 1));
            Some([
                iter.next()?,
                iter.next()?,
                iter.next()?,
                {
                    (0..self.columns - 3).for_each(|_| {
                        iter.next();
                    });
                    iter.next()?
                },
                iter.next()?,
                iter.next()?,
                {
                    (0..self.columns - 3).for_each(|_| {
                        iter.next();
                    });
                    iter.next()?
                },
                iter.next()?,
                iter.next()?,
            ])
        }
    }
    pub fn position<P: FnMut(I::Item) -> bool>(mut self, pred: P) -> Option<(usize, usize)> {
        self.inner
            .position(pred)
            .map(|flat| index_from_flat(self.columns, flat))
    }

    ///```rust
    ///
    /// // . . 1
    /// // . 2 .
    /// // 3 . .
    ///
    /// use grid_iter::IntoGridIter;
    /// (0..25).into_grid_iter(5)
    ///     .iter_diag_fwd(0,1)
    ///     .zip([1,5])
    ///     .for_each(|(l, r)| assert!(l == r));
    /// (0..25).into_grid_iter(5)
    ///     .iter_diag_fwd(3,2)
    ///     .zip([9,13,17,21])
    ///     .for_each(|(l, r)| assert!(l == r));
    /// (0..25).into_grid_iter(5)
    ///     .iter_diag_fwd(1,0)
    ///     .zip([1,5])
    ///     .for_each(|(l, r)| assert!(l == r));
    ///```
    pub fn iter_diag_fwd(self, col: usize, row: usize) -> DiagFwdIter<I> {
        let col_max = self.columns - 1;
        let (skip, take) = if col + row > self.columns - 1 {
            // lower right part
            (
                index_to_flat(self.columns, col_max, row - (col_max - col)),
                self.columns,
            )
        } else {
            // upper left part
            (index_to_flat(self.columns, row + col, 0), row + col + 1)
        };
        self.inner.skip(skip).step_by(col_max).take(take)
    }
    ///```rust
    ///
    /// // x . .
    /// // . x .
    /// // . . x
    ///
    /// use grid_iter::IntoGridIter;
    /// (0..25).into_grid_iter(5)
    ///     .iter_diag_bwd(1,2)
    ///     .zip([5,11,17,23])
    ///     .for_each(|(l, r)| assert!(l == r));
    /// (0..25).into_grid_iter(5)
    ///     .iter_diag_bwd(4,2)
    ///     .zip([2,8,14])
    ///     .for_each(|(l, r)| assert!(l == r));
    /// (0..25).into_grid_iter(5)
    ///     .iter_diag_bwd(4,0)
    ///     .zip([4])
    ///     .for_each(|(l, r)| assert!(l == r));
    ///```
    pub fn iter_diag_bwd(self, col: usize, row: usize) -> DiagBwdIter<I> {
        let diff = col.abs_diff(row);
        let (skip, take) = if col > row {
            //topright
            (index_to_flat(self.columns, diff, 0), self.columns - diff)
        } else {
            // botleft
            (index_to_flat(self.columns, 0, diff), self.columns)
        };
        self.inner.skip(skip).step_by(self.columns + 1).take(take)
    }
}
impl<I: Iterator<Item = T> + Clone, T> GridIter<I, T> {
    /// calculates the rows and tries to cache them
    /// for performance reasons you should use this function before using any other that needs the row count
    fn calc_rows(&mut self) -> usize {
        if let Some(rows) = self.rows {
            rows
        } else {
            assert!(self.columns != 0);
            let rows = self.clone().inner.count().div_ceil(self.columns);
            self.rows = Some(rows);
            rows
        }
    }
    ///```rust
    ///
    /// use grid_iter::IntoGridIter;
    /// assert!((0..25).into_grid_iter(5)
    ///     .iter_rows()
    ///     .flatten()
    ///     .sum::<usize>()
    ///     .eq(&(0..25).sum::<usize>()))
    ///```
    pub fn iter_rows(mut self) -> impl Iterator<Item = RowIter<I>> {
        let rows = self.calc_rows();
        repeat(self)
            .enumerate()
            .take(rows)
            .map(|(r, s)| s.iter_row(r))
    }
    ///```rust
    ///
    /// use grid_iter::IntoGridIter;
    /// assert!((0..25).into_grid_iter(5)
    ///     .iter_cols()
    ///     .flatten()
    ///     .sum::<usize>()
    ///     .eq(&(0..25).sum::<usize>()))
    ///```
    pub fn iter_cols(self) -> impl Iterator<Item = ColIter<I>> {
        let columns = self.columns;
        repeat(self)
            .enumerate()
            .take(columns)
            .map(|(c, s)| s.iter_col(c))
    }
    ///```rust
    ///
    /// use grid_iter::IntoGridIter;
    /// assert!((0..25).into_grid_iter(5)
    ///     .iter_diags_bwd()
    ///     .flatten()
    ///     .sum::<usize>()
    ///     .eq(&(0..25).sum::<usize>()))
    ///```
    pub fn iter_diags_bwd(mut self) -> impl Iterator<Item = DiagBwdIter<I>> {
        let rows = self.calc_rows();
        (0..self.columns)
            .rev()
            .zip(repeat(self.clone()))
            .map(|(c, s)| s.iter_diag_bwd(c, 0))
            .chain(
                (1..rows)
                    .zip(repeat(self))
                    .map(|(r, s)| s.iter_diag_bwd(0, r)),
            )
    }
    ///```rust
    ///
    /// use grid_iter::IntoGridIter;
    /// assert!((0..25).into_grid_iter(5)
    ///     .iter_diags_fwd()
    ///     .flatten()
    ///     .sum::<usize>()
    ///     .eq(&(0..25).sum::<usize>()))
    ///```
    pub fn iter_diags_fwd(mut self) -> impl Iterator<Item = DiagFwdIter<I>> {
        let rows = self.calc_rows();
        let col_max = self.columns - 1;
        (0..self.columns)
            .zip(repeat(self.clone()))
            .map(|(c, s)| s.iter_diag_fwd(c, 0))
            .chain(
                (1..rows)
                    .zip(repeat(self))
                    .map(move |(r, s)| s.iter_diag_fwd(col_max, r)),
            )
    }
    pub fn iter_kernels(mut self) -> impl Iterator<Item = [T; 9]> {
        let row_max = self.calc_rows() - 1;
        let col_max = self.columns - 1;
        repeat(self).zip(1..row_max).flat_map(move |(iter, r)| {
            (1..col_max).filter_map(move |c| iter.clone().get_kernel(c, r))
        })
    }
}
fn index_from_flat(gridcolumns: usize, flat: usize) -> (usize, usize) {
    assert!(gridcolumns != 0, "Columns set to 0! Cant calculate index");
    (flat % gridcolumns, flat / gridcolumns)
}
fn index_to_flat(gridcolumns: usize, col: usize, row: usize) -> usize {
    gridcolumns * row + col
}

// #[cfg(test)]
// mod tests {
//     use super::IntoGridIter;
//     #[test]
//     fn test_get() {
//         println!("{}", (0..25).into_grid_iter(5));
//         let t = (0..25)
//             .into_grid_iter(5)
//             .iter_kernels()
//             .for_each(|f| println!("{:?}", f));
//         println!("{t:?}");
//     }
// }
