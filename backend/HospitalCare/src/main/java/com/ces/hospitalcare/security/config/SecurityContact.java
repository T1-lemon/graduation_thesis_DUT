package com.ces.hospitalcare.security.config;

public class SecurityContact {
    public static final long EXPIRATION_TIME = 900000; // 15 minutes = 900000

    public static final long REFRESH_EXPIRATION_TIME = 864000000; // 10 days

    public static final long VERIFY_TOKEN_EXPIRATION_TIME = 300000; // 5 minutes
}