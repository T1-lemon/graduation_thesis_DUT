package com.ces.hospitalcare.dto;
import com.ces.hospitalcare.util.Role;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class UserDTO extends BaseDTO {
  private String firstName;

  private String lastName;

  private String profilePicture;

  private Integer age;

  private String email;

  private String address;

  private String phoneNumber;

  private Boolean gender;

  private Role role;
}
