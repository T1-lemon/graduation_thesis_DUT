package com.ces.hospitalcare.builder;
import com.ces.hospitalcare.entity.UserEntity;
import com.ces.hospitalcare.http.request.DoctorRegisterRequest;
import com.ces.hospitalcare.http.request.DoctorUpdateRequest;
import com.ces.hospitalcare.http.request.RegisterRequest;
import com.ces.hospitalcare.http.request.UpdateUserProfileRequest;
import org.springframework.stereotype.Component;

@Component
public class UserBuilder {
  public UserEntity userEntityBuild(RegisterRequest request, String password) {
    return UserEntity.builder()
        .firstName(request.getFirstName())
        .lastName(request.getLastName())
        .password(password)
        .email(request.getEmail())
        .role(request.getRole())
        .build();
  }

  public UserEntity userEntityUpdateBuild(UpdateUserProfileRequest request, UserEntity userEntity) {
    userEntity.setFirstName(request.getFirstName());
    userEntity.setLastName(request.getLastName());
    userEntity.setGender(request.getGender());
    userEntity.setPhoneNumber(request.getPhoneNumber());
    userEntity.setAddress(request.getAddress());
    return userEntity;
  }

  public UserEntity doctorEntityUpdateBuild(DoctorUpdateRequest request, UserEntity userEntity) {
    userEntity.setFirstName(request.getFirstName());
    userEntity.setLastName(request.getLastName());
    userEntity.setGender(request.getGender());
    userEntity.setPhoneNumber(request.getPhoneNumber());
    userEntity.setAddress(request.getAddress());
    userEntity.setAge(request.getAge());
    return userEntity;
  }

  public UserEntity doctorEntityBuild(DoctorRegisterRequest request, String password, String profilePicture) {
    return UserEntity.builder()
        .firstName(request.getFirstName())
        .lastName(request.getLastName())
        .password(password)
        .email(request.getEmail())
        .age(request.getAge())
        .address(request.getAddress())
        .gender(request.getGender())
        .phoneNumber(request.getPhoneNumber())
        .role(request.getRole())
        .profilePicture(profilePicture)
        .build();
  }
}
