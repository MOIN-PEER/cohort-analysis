-- Revenue per acquired customer in their first 12 months, and acquisition cost, by channel.
-- Only cohorts with a full 12 months of history (signed up by 2024-12) are included.

WITH first_year AS (
    SELECT
        c.channel,
        c.customer_id,
        COALESCE(SUM(o.revenue), 0) AS revenue_12m
    FROM customers c
    LEFT JOIN orders o
           ON o.customer_id = c.customer_id
          AND (CAST(substr(o.order_month, 1, 4) AS INTEGER) - CAST(substr(c.signup_month, 1, 4) AS INTEGER)) * 12
            + (CAST(substr(o.order_month, 6, 2) AS INTEGER) - CAST(substr(c.signup_month, 6, 2) AS INTEGER)) <= 12
    WHERE c.signup_month <= '2024-12'
    GROUP BY c.channel, c.customer_id
),
cac AS (
    SELECT channel, SUM(spend) * 1.0 / SUM(new_customers) AS cac
    FROM marketing_spend
    GROUP BY channel
)
SELECT
    f.channel,
    COUNT(*)                          AS customers,
    ROUND(AVG(f.revenue_12m), 2)      AS revenue_per_customer_12m,
    ROUND(k.cac, 2)                   AS cac,
    ROUND(AVG(f.revenue_12m) / k.cac, 2) AS revenue_to_cac
FROM first_year f
JOIN cac k ON k.channel = f.channel
GROUP BY f.channel
ORDER BY revenue_to_cac DESC;
