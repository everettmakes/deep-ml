-- your query
WITH total_profits AS (
    SELECT
        company,
        SUM(profit) AS total_profit
    FROM sales
    GROUP BY company
), rankings AS (
    SELECT
        company,
        total_profit,
        DENSE_RANK() OVER (ORDER BY total_profit DESC) AS ranking
    FROM
        total_profits
)   

SELECT
    company,
    total_profit
FROM rankings
WHERE ranking <= 3
ORDER BY total_profit DESC, company