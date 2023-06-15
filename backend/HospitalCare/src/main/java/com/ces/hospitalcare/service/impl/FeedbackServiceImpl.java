package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.dto.FeedbackDTO;
import com.ces.hospitalcare.entity.FeedbackEntity;
import com.ces.hospitalcare.entity.MedicalExaminationEntity;
import com.ces.hospitalcare.entity.UserEntity;
import com.ces.hospitalcare.http.request.FeedbackRequest;
import com.ces.hospitalcare.repository.FeedbackRepository;
import com.ces.hospitalcare.repository.MedicalExaminationRepository;
import com.ces.hospitalcare.repository.UserRepository;
import com.ces.hospitalcare.service.IFeedbackService;
import java.util.ArrayList;
import java.util.List;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class FeedbackServiceImpl implements IFeedbackService {
  @Autowired
  private FeedbackRepository feedbackRepository;

  @Autowired
  private UserRepository userRepository;

  @Autowired
  private MedicalExaminationRepository medicalExaminationRepository;

  @Autowired
  private ModelMapper mapper;

  public List<FeedbackDTO> createListFeedbackResponse(
      List<FeedbackEntity> listFeedbackEntity) {
    List<FeedbackDTO> listFeedbackDTO = new ArrayList<>();

    for (FeedbackEntity entity : listFeedbackEntity) {
      FeedbackDTO feedbackDTO = mapper.map(entity, FeedbackDTO.class);

      listFeedbackDTO.add(feedbackDTO);
    }

    return listFeedbackDTO;
  }

  @Override
  public List<FeedbackDTO> getAllFeedbackByExaminationId(Long examinationId) {
    List<FeedbackEntity> listFeedbackEntity = feedbackRepository.getAllByMedicalExaminationIdOrderByCreatedDateDesc(
        examinationId);

    return createListFeedbackResponse(listFeedbackEntity);
  }

  @Override
  public FeedbackDTO addFeedbackByPatient(FeedbackRequest feedbackRequest) {
    UserEntity patient = userRepository.getReferenceById(feedbackRequest.getPatientId());
    MedicalExaminationEntity medicalExaminationEntity = medicalExaminationRepository.getReferenceById(
        feedbackRequest.getExaminationId());

    FeedbackEntity feedbackEntity = feedbackRepository.save(
        FeedbackEntity.builder().medicalExamination(medicalExaminationEntity).patient(patient)
            .commentText(feedbackRequest.getCommentText()).build());
    return mapper.map(feedbackEntity, FeedbackDTO.class);
  }
}
