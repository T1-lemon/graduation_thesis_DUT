package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.builder.AppointmentBuilder;
import com.ces.hospitalcare.dto.AppointmentDTO;
import com.ces.hospitalcare.entity.AppointmentEntity;
import com.ces.hospitalcare.entity.TimeSlotEntity;
import com.ces.hospitalcare.entity.UserEntity;
import com.ces.hospitalcare.http.request.AppointmentRequest;
import com.ces.hospitalcare.http.response.AppointmentResponse;
import com.ces.hospitalcare.repository.AppointmentRepository;
import com.ces.hospitalcare.repository.TimeSlotRepository;
import com.ces.hospitalcare.repository.UserRepository;
import com.ces.hospitalcare.service.IAppointmentService;
import java.util.ArrayList;
import java.util.List;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Pageable;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

@Service
public class AppointmentServiceImpl implements IAppointmentService {
  @Autowired
  private UserRepository userRepository;

  @Autowired
  private AppointmentRepository appointmentRepository;

  @Autowired
  private AppointmentBuilder appointmentBuilder;

  @Autowired
  private TimeSlotRepository timeSlotRepository;

  @Autowired
  private ModelMapper mapper;

  @Override
  public int countAppointment() {
    return (int) appointmentRepository.count();
  }

  @Override
  public String cancelAppointmentMedicalArchive(Long doctorId) {
    List<AppointmentEntity> listAppointmentEntity = appointmentRepository.getAppointmentPendingOfMedicalArchive(
        doctorId);
    for (AppointmentEntity entity : listAppointmentEntity) {
      AppointmentDTO appointmentDTO = mapper.map(entity, AppointmentDTO.class);
      appointmentDTO.setStatus(2);
      changeStatusByDoctor(appointmentDTO, doctorId);
    }
    return "cancel successfully";
  }

  public List<AppointmentResponse> getListAppointmentOfPatient() {
    String userEmail = SecurityContextHolder.getContext().getAuthentication().getName();
    UserEntity user = userRepository.findByEmail(userEmail).orElseThrow();

    List<AppointmentEntity> listAppointmentEntity = appointmentRepository.getAllByPatientIdOrderByModifiedDateDesc(
        user.getId());

    List<AppointmentResponse> listAppointmentResponse = new ArrayList<>();

    for (AppointmentEntity entity : listAppointmentEntity) {
      AppointmentResponse appointmentResponse = appointmentBuilder.appointmentResponseBuilder(
          entity);
      listAppointmentResponse.add(appointmentResponse);
    }

    return listAppointmentResponse;
  }

  public List<AppointmentDTO> createAppointmentDTO(
      List<AppointmentEntity> listAppointmentEntity) {
    List<AppointmentDTO> listAppointmentDTO = new ArrayList<>();

    for (AppointmentEntity entity : listAppointmentEntity) {

      AppointmentDTO dto = mapper.map(entity, AppointmentDTO.class);

      listAppointmentDTO.add(dto);
    }

    return listAppointmentDTO;
  }

  @Override
  public int countByStatusAndDoctorId(int status, Long doctorId) {
    return appointmentRepository.countByStatusAndDoctorId(status, doctorId);
  }

  @Override
  public List<AppointmentDTO> getAllAppointmentOfDoctorPageableByStatusAndDoctorId(
      int appointmentStatus, Long doctorId, Pageable pageable) {
    List<AppointmentEntity> listAppointmentEntity = appointmentRepository.findAllByStatusAndDoctorIdOrderByModifiedDateDesc(
        appointmentStatus, doctorId, pageable);

    return createAppointmentDTO(listAppointmentEntity);
  }

  @Override
  public List<AppointmentDTO> getAllAppointmentPageable(Pageable pageable) {

    List<AppointmentEntity> listAppointmentEntity = appointmentRepository.findAll(pageable)
        .getContent();

    return createAppointmentDTO(listAppointmentEntity);
  }

  @Override
  public AppointmentDTO getDetailAppointment(Long appointmentId) {
    AppointmentEntity appointmentEntity = appointmentRepository.getReferenceById(appointmentId);
    return mapper.map(appointmentEntity, AppointmentDTO.class);
  }

  @Override
  public List<AppointmentDTO> getAllByDoctorIdAndPatientId(Long doctorId, Long patientId) {

    return createAppointmentDTO(
        appointmentRepository.getAllByDoctorIdAndPatientIdOrderByModifiedDateDesc(doctorId,
            patientId));
  }

  @Override
  public AppointmentDTO changeStatusByDoctor(AppointmentDTO appointmentChangeStatusDTO,
      Long doctorId) {
    AppointmentEntity appointmentEntityOld = appointmentRepository.getByIdAndDoctorId(
        appointmentChangeStatusDTO.getId(), doctorId);
    appointmentEntityOld.setStatus(appointmentChangeStatusDTO.getStatus());
    appointmentRepository.save(appointmentEntityOld);
    return mapper.map(appointmentEntityOld, AppointmentDTO.class);
  }

  @Override
  public AppointmentDTO bookAppointmentByPatient(AppointmentRequest appointmentRequest) {
    UserEntity patient = userRepository.getReferenceById(appointmentRequest.getPatientId());
    UserEntity doctor = userRepository.getReferenceById(appointmentRequest.getDoctorId());
    TimeSlotEntity timeSlotEntity = timeSlotRepository.getReferenceById(
        appointmentRequest.getTimeSlotId());

    AppointmentEntity appointmentEntity = appointmentRepository.save(
        AppointmentEntity.builder().patient(patient)
            .doctor(doctor).timeSlot(timeSlotEntity).build());

    return mapper.map(appointmentEntity, AppointmentDTO.class);
  }
}
