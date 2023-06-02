package com.ces.hospitalcare.http.response;
import com.ces.hospitalcare.dto.UserDTO;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class UserResponse {
  private UserDTO user;

  private int status;
}
