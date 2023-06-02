package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.dto.TimeSlotDTO;
import com.ces.hospitalcare.entity.MedicalExaminationEntity;
import com.ces.hospitalcare.entity.TimeSlotEntity;
import com.ces.hospitalcare.entity.UserEntity;
import com.ces.hospitalcare.http.request.TimeSlotRequest;
import com.ces.hospitalcare.http.response.TimeSlotResponse;
import com.ces.hospitalcare.repository.MedicalExaminationRepository;
import com.ces.hospitalcare.repository.TimeSlotRepository;
import com.ces.hospitalcare.repository.UserRepository;
import com.ces.hospitalcare.service.ITimeSlotService;
import java.util.ArrayList;
import java.util.List;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class TimeSlotServiceImpl implements ITimeSlotService {
  @Autowired
  private TimeSlotRepository timeSlotRepository;

  @Autowired
  private UserRepository userRepository;

  @Autowired
  private MedicalExaminationRepository medicalExaminationRepository;

  @Autowired
  private ModelMapper mapper;

  public List<TimeSlotResponse> createListTimeSlotResponse(
      List<TimeSlotEntity> listTimeSlotEntity) {
    List<TimeSlotResponse> listTimeSlotResponse = new ArrayList<>();

    for (TimeSlotEntity entity : listTimeSlotEntity) {

      TimeSlotDTO dto = mapper.map(entity, TimeSlotDTO.class);

      TimeSlotResponse timeSlotResponse =
          entity.getAppointment() != null ? TimeSlotResponse.builder().timeSlotDTO(dto)
              .doctorId(entity.getDoctor().getId()).appointmentId(entity.getAppointment().getId())
              .build() : TimeSlotResponse.builder().timeSlotDTO(dto)
              .doctorId(entity.getDoctor().getId()).build();

      listTimeSlotResponse.add(timeSlotResponse);
    }

    return listTimeSlotResponse;
  }

  @Override
  public List<TimeSlotResponse> getAllTimeSlotByExaminationId(Long examinationId) {
    List<TimeSlotEntity> listTimeSlotEntity = timeSlotRepository.getAllByMedicalExaminationId(
        examinationId);

    return createListTimeSlotResponse(listTimeSlotEntity);
  }

  @Override
  public TimeSlotResponse getDetailTimeSlot(Long id) {
    TimeSlotEntity timeSlotEntity = timeSlotRepository.getReferenceById(id);
    TimeSlotDTO timeSlotDTO = mapper.map(timeSlotEntity, TimeSlotDTO.class);
    if (timeSlotEntity.getAppointment() != null) {
      return TimeSlotResponse.builder().timeSlotDTO(timeSlotDTO)
          .doctorId(timeSlotEntity.getDoctor().getId())
          .appointmentId(timeSlotEntity.getAppointment().getId()).build();
    }

    return TimeSlotResponse.builder().timeSlotDTO(timeSlotDTO)
        .doctorId(timeSlotEntity.getDoctor().getId())
        .build();
  }

  @Override
  public List<TimeSlotResponse> getAllTimeSlotsOfCurrentWeek(Long doctorId) {
    List<TimeSlotEntity> listTimeSlotEntity = timeSlotRepository.getAllTimeSlotsOfCurrentWeek(
        doctorId);
    return createListTimeSlotResponse(listTimeSlotEntity);
  }

  @Override
  public List<TimeSlotResponse> addAllTimeSlotsOfDoctor(List<TimeSlotRequest> listTimeSlotRequest,
      Long doctorId) {
    UserEntity doctorEntity = userRepository.getReferenceById(doctorId);
    MedicalExaminationEntity medicalExaminationEntity = medicalExaminationRepository.getByDoctorId(
        doctorId);
    List<TimeSlotEntity> listTimeSlotEntity = new ArrayList<>();

    for (TimeSlotRequest timeSlotRequest : listTimeSlotRequest) {
      TimeSlotEntity timeSlotEntity = TimeSlotEntity.builder()
          .startTime(timeSlotRequest.getStartTime()).duration(
              timeSlotRequest.getDuration()).doctor(doctorEntity)
          .medicalExamination(medicalExaminationEntity).build();
      listTimeSlotEntity.add(timeSlotEntity);
    }

    timeSlotRepository.saveAll(listTimeSlotEntity);
    return createListTimeSlotResponse(listTimeSlotEntity);
  }

  @Override
  public TimeSlotDTO updateTimeSlotByDoctor(TimeSlotDTO timeSlotDTO, Long doctorId) {
    TimeSlotEntity timeSlotEntityOld = timeSlotRepository.getByIdAndDoctorId(timeSlotDTO.getId(),
        doctorId);
    timeSlotEntityOld.setStartTime(timeSlotDTO.getStartTime());
    timeSlotEntityOld.setDuration(timeSlotDTO.getDuration());
    timeSlotRepository.save(timeSlotEntityOld);
    return mapper.map(timeSlotEntityOld, TimeSlotDTO.class);
  }

  @Override
  public String deleteTimeSlotByDoctor(Long timeSlotID) {
    timeSlotRepository.deleteById(timeSlotID);
    return "Delete successful";
  }
}
