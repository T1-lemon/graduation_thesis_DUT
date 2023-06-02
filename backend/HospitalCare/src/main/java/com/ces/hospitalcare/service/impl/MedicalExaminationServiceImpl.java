package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.builder.MedicalExaminationBuilder;
import com.ces.hospitalcare.dto.MedicalExaminationDTO;
import com.ces.hospitalcare.entity.MedicalExaminationEntity;
import com.ces.hospitalcare.http.request.MedicalExaminationRequest;
import com.ces.hospitalcare.http.response.MedicalExaminationResponse;
import com.ces.hospitalcare.http.response.TimeSlotResponse;
import com.ces.hospitalcare.repository.DepartmentRepository;
import com.ces.hospitalcare.repository.MedicalExaminationRepository;
import com.ces.hospitalcare.repository.UserRepository;
import com.ces.hospitalcare.service.IAppointmentService;
import com.ces.hospitalcare.service.IMedicalExaminationService;
import com.ces.hospitalcare.service.ITimeSlotService;
import java.util.ArrayList;
import java.util.List;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class MedicalExaminationServiceImpl implements IMedicalExaminationService {
  @Autowired
  private MedicalExaminationRepository medicalExaminationRepository;

  @Autowired
  private ITimeSlotService timeSlotService;

  @Autowired
  private IAppointmentService appointmentService;

  @Autowired
  private UserRepository userRepository;

  @Autowired
  private DepartmentRepository departmentRepository;

  @Autowired
  private ModelMapper mapper;

  @Autowired
  private MedicalExaminationBuilder medicalExaminationBuilder;

  public List<MedicalExaminationResponse> createListMedicalExaminationResponse(
      List<MedicalExaminationEntity> listMedicalExaminationEntity) {
    List<MedicalExaminationResponse> listMedicalExaminationResponse = new ArrayList<>();

    for (MedicalExaminationEntity entity : listMedicalExaminationEntity) {
      MedicalExaminationDTO dto = mapper.map(entity, MedicalExaminationDTO.class);

      List<TimeSlotResponse> listTimeSlotDTO =
          timeSlotService.getAllTimeSlotByExaminationId(entity.getId());

      MedicalExaminationResponse medicalExaminationResponse = MedicalExaminationResponse.builder()
          .medicalExamination(dto).listTimeSlot(listTimeSlotDTO)
          .build();
      listMedicalExaminationResponse.add(medicalExaminationResponse);
    }

    return listMedicalExaminationResponse;
  }

  @Override
  public List<MedicalExaminationResponse> getAllMedicalExamination() {
    List<MedicalExaminationEntity> listMedicalExaminationEntity = medicalExaminationRepository.getAllByStatusArchive(
        0);

    return createListMedicalExaminationResponse(listMedicalExaminationEntity);
  }

  @Override
  public MedicalExaminationResponse getDetailMedicalExamination(Long id) {

    MedicalExaminationEntity medicalExaminationEntity = medicalExaminationRepository.getReferenceById(
        id);
    MedicalExaminationDTO dto = mapper.map(medicalExaminationEntity, MedicalExaminationDTO.class);

    List<TimeSlotResponse> listTimeSlot =
        timeSlotService.getAllTimeSlotByExaminationId(medicalExaminationEntity.getId());

    return MedicalExaminationResponse.builder()
        .medicalExamination(dto).listTimeSlot(listTimeSlot)
        .build();
  }

  public List<MedicalExaminationEntity> filterMedicalExaminationEntity(
      Long minPrice, Long maxPrice, String[] categories) {

    if (minPrice == null && maxPrice == null) {
      List<MedicalExaminationEntity> listMedicalExaminationEntity =
          categories.length == 0 ? medicalExaminationRepository.findAll() :
              medicalExaminationRepository.filterMedicalExaminationByCategory(categories);

      return listMedicalExaminationEntity;
    }

    if (categories.length == 0) {

      return medicalExaminationRepository.filterMedicalExaminationByPrice(minPrice, maxPrice);
    }

    return medicalExaminationRepository.filterMedicalExaminationByCategoryAndPrice(
        minPrice, maxPrice, categories);
  }

  @Override
  public List<MedicalExaminationResponse> filterMedicalExaminationByCategoryAndPrice(
      Long minPrice, Long maxPrice, String[] categories) {

    List<MedicalExaminationEntity> listMedicalExaminationEntity =
        filterMedicalExaminationEntity(minPrice, maxPrice, categories);

    return createListMedicalExaminationResponse(listMedicalExaminationEntity);
  }

  @Override
  public MedicalExaminationDTO addMedicalExamination(
      MedicalExaminationRequest medicalExaminationRequest) {
    MedicalExaminationEntity medicalExaminationEntity = mapper.map(medicalExaminationRequest,
        MedicalExaminationEntity.class);
    medicalExaminationEntity.setDoctor(
        userRepository.getReferenceById(medicalExaminationRequest.getDoctorId()));
    medicalExaminationEntity.setDepartment(
        departmentRepository.getReferenceById(medicalExaminationRequest.getDepartmentId()));
    medicalExaminationRepository.save(medicalExaminationEntity);

    return mapper.map(medicalExaminationEntity, MedicalExaminationDTO.class);
  }

  @Override
  public MedicalExaminationDTO updateMedicalExamination(
      MedicalExaminationRequest medicalExaminationRequest, Long medicalId) {
    MedicalExaminationEntity medicalExaminationEntity = medicalExaminationRepository.getReferenceById(
        medicalId);
    medicalExaminationRepository.save(
        medicalExaminationBuilder.medicalUpdateBuild(medicalExaminationRequest,
            medicalExaminationEntity));

    return mapper.map(medicalExaminationEntity, MedicalExaminationDTO.class);
  }

  @Override
  public MedicalExaminationDTO archiveMedicalExamination(
      MedicalExaminationDTO medicalExaminationDTO) {
    MedicalExaminationEntity medicalExaminationEntityOld = medicalExaminationRepository.getReferenceById(
        medicalExaminationDTO.getId());
    medicalExaminationEntityOld.setStatusArchive(medicalExaminationDTO.getStatusArchive());
    medicalExaminationRepository.save(medicalExaminationEntityOld);
    appointmentService.cancelAppointmentMedicalArchive(
        medicalExaminationEntityOld.getDoctor().getId());
    return mapper.map(medicalExaminationEntityOld, MedicalExaminationDTO.class);
  }
}
