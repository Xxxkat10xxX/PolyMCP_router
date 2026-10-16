CREATE TABLE IF NOT EXISTS system_incidents (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    error_message TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO system_incidents (service_name, error_message, status) VALUES
('auth_service', 'Token expired unexpectedly for user_id 401', 'open'),
('billing_service', 'Payment gateway timeout on transaction 8812', 'investigating'),
('api_gateway', '502 Bad Gateway to upstream order_service', 'open');