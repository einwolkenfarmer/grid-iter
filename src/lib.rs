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
            columns: self.columns,
            rows: self.rows,
        }
    }
}
impl<I: Iterator<Item = T> + Clone, T: core::fmt::Display> core::fmt::Display for GridIter<I, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.clone().iter_rows().for_each(|col| {
            col.for_each(|ch| write!(f, "{}\t", ch).unwrap());
            writeln!(f).unwrap();
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
    pub fn iter_col(self, column_index: usize) -> ColIter<I> {
        assert!(column_index < self.columns);
        self.inner.skip(column_index).step_by(self.columns)
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
    pub fn iter_row(self, row_index: usize) -> RowIter<I> {
        self.inner.skip(row_index * self.columns).take(self.columns)
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
    pub fn get(self, col_index: usize, row_index: usize) -> Option<<I as IntoIterator>::Item> {
        self.inner
            .skip(grid_index_to_flat(self.columns, col_index, row_index))
            .take(1)
            .next()
    }
    pub fn get_kernel(
        self,
        column_index: usize,
        row_index: usize,
    ) -> Option<[<I as IntoIterator>::Item; 9]> {
        if column_index == 0 || column_index == self.columns - 1 || row_index == 0 {
            None
        } else {
            let mut iter = self.inner.skip(grid_index_to_flat(
                self.columns,
                column_index - 1,
                row_index - 1,
            ));
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
            .map(|flat| flat_index_to_grid(self.columns, flat))
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
    pub fn iter_diag_fwd(self, column_index: usize, row_index: usize) -> DiagFwdIter<I> {
        let col_max = self.columns - 1;
        let (skip, take) = if column_index + row_index > self.columns - 1 {
            // lower right part
            (
                grid_index_to_flat(self.columns, col_max, row_index - (col_max - column_index)),
                self.columns,
            )
        } else {
            // upper left part
            (
                grid_index_to_flat(self.columns, row_index + column_index, 0),
                row_index + column_index + 1,
            )
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
    pub fn iter_diag_bwd(self, column_index: usize, row_index: usize) -> DiagBwdIter<I> {
        let diff = column_index.abs_diff(row_index);
        let (skip, take) = if column_index > row_index {
            //topright
            (
                grid_index_to_flat(self.columns, diff, 0),
                self.columns - diff,
            )
        } else {
            // botleft
            (grid_index_to_flat(self.columns, 0, diff), self.columns)
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
fn flat_index_to_grid(grid_columns: usize, flat_index: usize) -> (usize, usize) {
    assert!(grid_columns != 0, "Columns set to 0! Cant calculate index");
    (flat_index % grid_columns, flat_index / grid_columns)
}
fn grid_index_to_flat(grid_columns: usize, column_index: usize, row_index: usize) -> usize {
    grid_columns * row_index + column_index
}

#[cfg(test)]
mod tests {
    use crate::{flat_index_to_grid, grid_index_to_flat};

    #[test]
    fn test_index() {
        let (grid_col, flat_index) = (5123, 55);
        let (col, row) = flat_index_to_grid(grid_col, flat_index);
        let flat_index2 = grid_index_to_flat(grid_col, col, row);
        assert_eq!(flat_index, flat_index2);
    }
}
