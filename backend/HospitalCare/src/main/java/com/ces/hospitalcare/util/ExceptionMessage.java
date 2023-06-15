package com.ces.hospitalcare.util;
public enum ExceptionMessage {
  USERNAME_PASSWORD_INVALIDATE("Username or Password Incorrect"),
  EMAIL_ALREADY_EXIST("Email Already Exist"),
  RESOURCE_NOT_FOUND("Resource Not Found"),
  INVALID_REFRESH_TOKEN("Invalid Refresh Token"),
  INVALID_ACCESS_TOKEN("Invalid Access Token"),
  INVALID_VERIFY_TOKEN("Invalid Verify Token");

  private String message;

  ExceptionMessage(String message) {
    this.message = message;
  }

  public String getMessage() {
    return message;
  }
}