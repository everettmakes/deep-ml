SELECT
    COALESCE(SUM(CASE WHEN department = 'Engineering' THEN salary END), 0) AS engineering_total,
    COALESCE(SUM(CASE WHEN department = 'Sales' THEN salary END), 0) AS sales_total,
    COALESCE(SUM(CASE WHEN department = 'Engineering' THEN salary END), 0)
  - COALESCE(SUM(CASE WHEN department = 'Sales' THEN salary END), 0) AS salary_difference
FROM employees;