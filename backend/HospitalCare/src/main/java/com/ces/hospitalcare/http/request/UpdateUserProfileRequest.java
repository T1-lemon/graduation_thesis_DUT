package com.ces.hospitalcare.http.request;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class UpdateUserProfileRequest {
  private String firstName;

  private String lastName;

  private String phoneNumber;

  private Boolean gender;

  private String address;

  private Integer age;
}
