WITH dept_avg AS (
    SELECT
        id,
        name,
        department,
        salary,
        AVG(salary) OVER (PARTITION BY department) AS avg_salary
    FROM employees
)
SELECT id, name, department, salary
FROM dept_avg
WHERE salary > avg_salary;