package com.ces.hospitalcare.http.request;
import com.ces.hospitalcare.util.Role;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class RegisterRequest {
  private String firstName;

  private String lastName;

  private String email;

  private String password;

  private Role role;

  @JsonCreator
  public RegisterRequest(String value) {
    this.role = Role.valueOf(value.toUpperCase());
  }

  @JsonValue
  public String getValue() {
    return role.name().toLowerCase();
  }
}
