    .text
    .globl bench_shuffle
bench_shuffle:
    shuffle.idx v8, v4, 3
    shuffle.up v9, v5, 4
    shuffle.down v10, v6, 7
    shuffle.bfly v11, v7, 15
    ret
