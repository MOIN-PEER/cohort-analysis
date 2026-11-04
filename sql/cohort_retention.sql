-- Cohort activity retention in SQL (SQLite).
-- For each signup cohort and each month since signup: the % of the cohort that received a box.
-- The Python version (cohorts.activity_retention) must give identical numbers; a test checks this.

WITH cohort AS (
    SELECT customer_id, signup_month
    FROM customers
),
cohort_size AS (
    SELECT signup_month, COUNT(*) AS customers
    FROM cohort
    GROUP BY signup_month
),
activity AS (
    SELECT
        c.signup_month,
        -- months between signup and order: (year difference * 12) + month difference
        (CAST(substr(o.order_month, 1, 4) AS INTEGER) - CAST(substr(c.signup_month, 1, 4) AS INTEGER)) * 12
          + (CAST(substr(o.order_month, 6, 2) AS INTEGER) - CAST(substr(c.signup_month, 6, 2) AS INTEGER)) AS age,
        o.customer_id
    FROM orders o
    JOIN cohort c ON c.customer_id = o.customer_id
)
SELECT
    a.signup_month AS cohort,
    a.age,
    COUNT(DISTINCT a.customer_id)                                         AS active_customers,
    s.customers                                                           AS cohort_size,
    ROUND(100.0 * COUNT(DISTINCT a.customer_id) / s.customers, 1)         AS retention_pct
FROM activity a
JOIN cohort_size s ON s.signup_month = a.signup_month
GROUP BY a.signup_month, a.age
ORDER BY a.signup_month, a.age;
