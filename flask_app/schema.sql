-- ─────────────────────────────────────────────────────────────────────────────
-- Schema MySQL untuk OCEAN Personality Prediction Web App
-- Engine: InnoDB (sesuai request client)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE DATABASE IF NOT EXISTS ocean_personality
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE ocean_personality;

-- ─── Users ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            INT UNSIGNED      NOT NULL AUTO_INCREMENT,
    username      VARCHAR(64)       NOT NULL UNIQUE,
    email         VARCHAR(128)      NOT NULL UNIQUE,
    password_hash VARCHAR(255)      NOT NULL,
    full_name     VARCHAR(128)      DEFAULT NULL,
    created_at    DATETIME          NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_username (username),
    INDEX idx_email    (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ─── Predictions ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                INT UNSIGNED   NOT NULL AUTO_INCREMENT,
    user_id           INT UNSIGNED   NOT NULL,
    subject_name      VARCHAR(128)   DEFAULT NULL COMMENT 'Nama orang yang divideo',
    video_filename    VARCHAR(255)   NOT NULL,
    source            ENUM('upload','record') NOT NULL DEFAULT 'upload',
    openness          DECIMAL(6,4)   NOT NULL COMMENT 'Skor OCEAN dalam rentang 0 - 100%',
    conscientiousness DECIMAL(6,4)   NOT NULL COMMENT 'Skor OCEAN dalam rentang 0 - 100%',
    extraversion      DECIMAL(6,4)   NOT NULL COMMENT 'Skor OCEAN dalam rentang 0 - 100%',
    agreeableness     DECIMAL(6,4)   NOT NULL COMMENT 'Skor OCEAN dalam rentang 0 - 100%',
    neuroticism       DECIMAL(6,4)   NOT NULL COMMENT 'Skor OCEAN dalam rentang 0 - 100%',
    processing_time   DECIMAL(8,3)   DEFAULT NULL COMMENT 'Detik',
    created_at        DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_user_id (user_id),
    INDEX idx_created (created_at),
    CONSTRAINT fk_predictions_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
