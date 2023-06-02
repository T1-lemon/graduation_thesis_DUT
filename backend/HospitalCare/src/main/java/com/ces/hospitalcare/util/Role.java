package com.ces.hospitalcare.util;
public enum Role {
  ADMIN("ADMIN"),
  PATIENT("PATIENT"),
  DOCTOR("DOCTOR");
  private String role;

  private Role(String role) {
    this.role = role;
  }
}