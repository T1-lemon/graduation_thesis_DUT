package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.builder.UserBuilder;
import com.ces.hospitalcare.dto.MedicalExaminationDTO;
import com.ces.hospitalcare.dto.UserDTO;
import com.ces.hospitalcare.entity.MedicalExaminationEntity;
import com.ces.hospitalcare.entity.UserEntity;
import com.ces.hospitalcare.http.exception.AlreadyExistException;
import com.ces.hospitalcare.http.exception.ResourceNotFoundException;
import com.ces.hospitalcare.http.request.DoctorRegisterRequest;
import com.ces.hospitalcare.http.request.DoctorRequest;
import com.ces.hospitalcare.http.request.DoctorUpdateRequest;
import com.ces.hospitalcare.http.request.MedicalExaminationRequest;
import com.ces.hospitalcare.http.request.UpdateUserProfileRequest;
import com.ces.hospitalcare.http.response.DoctorResponse;
import com.ces.hospitalcare.http.response.UserResponse;
import com.ces.hospitalcare.repository.DepartmentRepository;
import com.ces.hospitalcare.repository.MedicalExaminationRepository;
import com.ces.hospitalcare.repository.UserRepository;
import com.ces.hospitalcare.service.IMedicalExaminationService;
import com.ces.hospitalcare.service.IUserService;
import com.ces.hospitalcare.util.Role;
import com.cloudinary.Cloudinary;
import com.cloudinary.utils.ObjectUtils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import lombok.NoArgsConstructor;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

@Service
@NoArgsConstructor
public class UserServiceImpl implements IUserService {
  @Autowired
  UserRepository userRepository;

  @Autowired
  private PasswordEncoder passwordEncoder;

  @Autowired
  private MedicalExaminationRepository medicalExaminationRepository;

  @Autowired
  private IMedicalExaminationService medicalExaminationService;

  @Autowired
  private ModelMapper mapper;

  @Autowired
  private Cloudinary cloudinary;

  @Autowired
  private UserBuilder userBuilder;

  @Autowired
  private DepartmentRepository departmentRepository;

  @Override
  public UserResponse getCurrentUser() {
    UserDetails userDetails = (UserDetails) SecurityContextHolder.getContext().getAuthentication()
        .getPrincipal();
    String userEmail = userDetails.getUsername();

    UserEntity user = userRepository.findByEmail(userEmail).orElseThrow();
    UserDTO userDTO = mapper.map(user, UserDTO.class);

    return UserResponse.builder()
        .status(HttpStatus.OK.value())
        .user(userDTO)
        .build();
  }

  @Override
  public UserDTO updateUserProfilePicture(Long userId, MultipartFile multipartFile)
      throws IOException {
    UserEntity user = userRepository.findById(userId)
        .orElseThrow(() -> new ResourceNotFoundException("User not found with id: " + userId));
    System.out.println(userId);
    Map response = cloudinary.uploader().upload(multipartFile.getBytes(), ObjectUtils.asMap(
        "public_id",
        "profilePicture" + "/" + userId + "/" + multipartFile.getName()
    ));
    System.out.println("upload success");
    String profilePictureUrl = (String) response.get("secure_url");
    user.setProfilePicture(profilePictureUrl);
    userRepository.save(user);

    UserDTO userDTO = mapper.map(user, UserDTO.class);
    return userDTO;
  }

  @Override
  public UserDTO updateUserProfile(Long userId, UpdateUserProfileRequest updateUserProfileRequest)
      throws IOException {
    UserEntity user = userRepository.findById(userId)
        .orElseThrow(() -> new ResourceNotFoundException("User not found with id: " + userId));

    UserEntity updatedUser = userBuilder.userEntityUpdateBuild(updateUserProfileRequest, user);
    userRepository.save(updatedUser);

    UserDTO userDTO = mapper.map(user, UserDTO.class);
    return userDTO;
  }

  public List<UserDTO> createListUserDTO(
      List<UserEntity> listUserEntity) {
    List<UserDTO> listUserDTO = new ArrayList<>();

    for (UserEntity entity : listUserEntity) {
      UserDTO dto = mapper.map(entity, UserDTO.class);

      listUserDTO.add(dto);
    }

    return listUserDTO;
  }

  @Override
  public List<UserDTO> getAllPatient() {
    List<UserEntity> listPatientEntity = userRepository.getAllByRole(Role.PATIENT);
    return createListUserDTO(listPatientEntity);
  }

  @Override
  public List<DoctorResponse> getAllDoctor(int statusArchive) {
    List<DoctorResponse> listDoctorResponse = new ArrayList<>();
    List<UserEntity> listDoctorEntity = userRepository.getAllByRole(Role.DOCTOR);
    for (UserEntity doctorEntity : listDoctorEntity) {
      MedicalExaminationEntity medicalExaminationEntity = medicalExaminationRepository.getByDoctorId(
          doctorEntity.getId());
      MedicalExaminationDTO medicalExaminationDTO = mapper.map(medicalExaminationEntity,
          MedicalExaminationDTO.class);
      UserDTO doctorDTO = mapper.map(doctorEntity, UserDTO.class);
      if (medicalExaminationEntity.getStatusArchive() == statusArchive) {
        DoctorResponse doctorResponse = DoctorResponse.builder().doctor(doctorDTO)
            .medicalExamination(medicalExaminationDTO).build();
        listDoctorResponse.add((doctorResponse));
      }
    }
    return listDoctorResponse;
  }

  @Override
  public UserDTO getDetailUser(Long userId) {
    return mapper.map(userRepository.getReferenceById(userId), UserDTO.class);
  }

  @Override
  public DoctorResponse getDetailDoctor(Long doctorId) {
    UserEntity doctorEntity = userRepository.getReferenceById(doctorId);
    MedicalExaminationEntity medicalExaminationEntity = medicalExaminationRepository.getByDoctorId(
        doctorId);
    MedicalExaminationDTO medicalExaminationDTO = mapper.map(medicalExaminationEntity,
        MedicalExaminationDTO.class);
    UserDTO doctorDTO = mapper.map(doctorEntity, UserDTO.class);
    DoctorResponse doctorResponse = DoctorResponse.builder().doctor(doctorDTO)
        .medicalExamination(medicalExaminationDTO).build();
    return doctorResponse;
  }

  @Override
  public DoctorResponse addDoctor(DoctorRequest doctorRequest, MultipartFile multipartFile)
      throws IOException {
    DoctorRegisterRequest doctorRegister = doctorRequest.getDoctor();
    MedicalExaminationRequest medicalExamination = doctorRequest.getMedicalExamination();
    if (userRepository.findByEmail(doctorRegister.getEmail()).isEmpty() == false) {
      throw new AlreadyExistException("Email already exists");
    }

    Map response = cloudinary.uploader().upload(multipartFile.getBytes(), ObjectUtils.asMap(
        "public_id",
        "profilePicture" + "/" + multipartFile.getName()
    ));

    String profilePictureUrl = (String) response.get("secure_url");

    UserEntity doctor = userBuilder.doctorEntityBuild(doctorRegister,
        passwordEncoder.encode(doctorRegister.getPassword()), profilePictureUrl);
    userRepository.save(doctor);
    medicalExamination.setDoctorId(doctor.getId());
    MedicalExaminationDTO medicalExaminationDTO = medicalExaminationService.addMedicalExamination(
        medicalExamination);

    return DoctorResponse.builder().doctor(mapper.map(doctor, UserDTO.class))
        .medicalExamination(medicalExaminationDTO).build();
  }

  @Override
  public String checkEmailDoctor(UserDTO doctorDTO) {
    if (userRepository.findByEmail(doctorDTO.getEmail()).isEmpty() == false) {
      throw new AlreadyExistException("Email already exists");
    }
    return "Valid email";
  }

  @Override
  public DoctorResponse updateProfileDoctor(DoctorUpdateRequest doctorUpdateRequest) {
    Long doctorId = doctorUpdateRequest.getDoctorId();
    UserEntity doctorEntityOld = userRepository.findByIdAndRole(doctorId, Role.DOCTOR)
        .orElseThrow(() -> new ResourceNotFoundException("Doctor not found with id: " + doctorId));

    UserEntity updatedDoctor = userBuilder.doctorEntityUpdateBuild(doctorUpdateRequest,
        doctorEntityOld);
    userRepository.save(updatedDoctor);

    MedicalExaminationEntity medicalExaminationEntity = medicalExaminationRepository.getByDoctorId(
        doctorId);
    medicalExaminationEntity.setDepartment(
        departmentRepository.getReferenceById(doctorUpdateRequest.getDepartmentId()));
    medicalExaminationRepository.save(medicalExaminationEntity);

    return DoctorResponse.builder().doctor(mapper.map(updatedDoctor, UserDTO.class))
        .medicalExamination(mapper.map(medicalExaminationEntity, MedicalExaminationDTO.class))
        .build();
  }
}
