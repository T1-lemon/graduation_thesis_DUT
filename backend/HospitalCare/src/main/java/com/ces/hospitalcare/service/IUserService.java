package com.ces.hospitalcare.service;
import com.ces.hospitalcare.dto.UserDTO;
import com.ces.hospitalcare.http.request.DoctorRequest;
import com.ces.hospitalcare.http.request.DoctorUpdateRequest;
import com.ces.hospitalcare.http.request.UpdateUserProfileRequest;
import com.ces.hospitalcare.http.response.DoctorResponse;
import com.ces.hospitalcare.http.response.UserResponse;
import java.io.IOException;
import java.util.List;
import org.springframework.web.multipart.MultipartFile;

public interface IUserService {
  UserResponse getCurrentUser();

  UserDTO updateUserProfilePicture(Long userId, MultipartFile multipartFile) throws IOException;

  UserDTO updateUserProfile(Long userId, UpdateUserProfileRequest updateUserProfileRequest)
      throws IOException;

  List<UserDTO> getAllPatient();

  List<DoctorResponse> getAllDoctor(int statusArchive);

  UserDTO getDetailUser(Long userId);

  DoctorResponse getDetailDoctor(Long doctorId);

  DoctorResponse addDoctor(DoctorRequest doctorRequest, MultipartFile multipartFile) throws IOException;

  String checkEmailDoctor(UserDTO doctorDTO);

  DoctorResponse updateProfileDoctor(DoctorUpdateRequest doctorUpdateRequest);
}
