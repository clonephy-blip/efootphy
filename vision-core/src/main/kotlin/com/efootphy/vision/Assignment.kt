package com.efootphy.vision

/** Global one-to-one assignment, with an independent unmatched option for every row. */
object Assignment {
    fun solve(cost: List<DoubleArray>, unmatched: Double): IntArray {
        require(unmatched.isFinite() && unmatched > 0)
        val n = cost.size
        if (n == 0) return IntArray(0)
        val real = cost[0].size
        require(cost.all { row -> row.size == real && row.all { !it.isNaN() && it >= 0 } })
        val m = real + n
        val u = DoubleArray(n + 1)
        val v = DoubleArray(m + 1)
        val p = IntArray(m + 1)
        val way = IntArray(m + 1)
        fun value(i: Int, j: Int): Double = if (j > real) unmatched else
            cost[i - 1][j - 1].let { if (it.isFinite()) it else 1e12 }
        for (i in 1..n) {
            p[0] = i
            var j0 = 0
            val minv = DoubleArray(m + 1) { Double.POSITIVE_INFINITY }
            val used = BooleanArray(m + 1)
            do {
                used[j0] = true
                val i0 = p[j0]
                var delta = Double.POSITIVE_INFINITY
                var j1 = 0
                for (j in 1..m) if (!used[j]) {
                    val cur = value(i0, j) - u[i0] - v[j]
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0 }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j }
                }
                for (j in 0..m) {
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta }
                    else minv[j] -= delta
                }
                j0 = j1
            } while (p[j0] != 0)
            do {
                val j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while (j0 != 0)
        }
        val result = IntArray(n) { -1 }
        for (j in 1..real) if (p[j] != 0 && value(p[j], j) < unmatched) result[p[j] - 1] = j - 1
        return result
    }
}
