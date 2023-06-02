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
public class DoctorRegisterRequest {
  private String firstName;

  private String lastName;

  private String email;

  private String password;

  private Integer age;

  private String address;

  private String phoneNumber;

  private Boolean gender;

  private Role role;

  @JsonCreator
  public DoctorRegisterRequest(String value) {
    this.role = Role.valueOf(value.toUpperCase());
  }

  @JsonValue
  public String getValue() {
    return role.name().toLowerCase();
  }
}
