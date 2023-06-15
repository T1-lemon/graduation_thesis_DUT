package com.ces.hospitalcare.security.service;
import com.ces.hospitalcare.entity.UserEntity;
import java.util.HashMap;
import java.util.Map;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
@Data
@NoArgsConstructor
@AllArgsConstructor
public class VerifyTokenService {
  @Autowired
  JwtService jwtService;
  private Map<String, String> verifyToken = new HashMap<>();

  private Map<String, UserEntity> verifyUserEntity = new HashMap<>();

  public void setVerifyToken(String token) {
    String userEmail = jwtService.extractUsername(token);
    verifyToken.put(userEmail, token);
  }

  public String getVerifyToken(String userEmail) {
    return verifyToken.get(userEmail);
  }

  public void setVerifyUserEntity(UserEntity userEntity) {
    verifyUserEntity.put(userEntity.getEmail(), userEntity);
  }

  public UserEntity getVerifyUserEntity(String userEmail) {
    return verifyUserEntity.get(userEmail);
  }
}
