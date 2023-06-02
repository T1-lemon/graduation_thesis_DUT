package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.dto.DepartmentDTO;
import com.ces.hospitalcare.entity.DepartmentEntity;
import com.ces.hospitalcare.entity.MedicalExaminationEntity;
import com.ces.hospitalcare.http.exception.DeleteDepartmentException;
import com.ces.hospitalcare.repository.DepartmentRepository;
import com.ces.hospitalcare.repository.MedicalExaminationRepository;
import com.ces.hospitalcare.service.IDepartmentService;
import com.cloudinary.Cloudinary;
import com.cloudinary.utils.ObjectUtils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

@Service
public class DepartmentServiceImpl implements IDepartmentService {
  @Autowired
  private DepartmentRepository departmentRepository;

  @Autowired
  private ModelMapper mapper;

  @Autowired
  private MedicalExaminationRepository medicalExaminationRepository;

  @Autowired
  private Cloudinary cloudinary;


  public List<DepartmentDTO> createListDepartmentDTO(List<DepartmentEntity> departmentEntityList) {

    List<DepartmentDTO> departmentDTOList = new ArrayList<>();

    for (DepartmentEntity entity : departmentEntityList) {
      DepartmentDTO dto = mapper.map(entity, DepartmentDTO.class);

      departmentDTOList.add(dto);
    }

    return departmentDTOList;
  }

  @Override
  public List<DepartmentDTO> getAllDepartment() {

    return createListDepartmentDTO(departmentRepository.findAll());
  }

  @Override
  public DepartmentDTO addDepartment(DepartmentDTO departmentDTO, MultipartFile multipartFile)
      throws IOException {

    Map response = cloudinary.uploader().upload(multipartFile.getBytes(), ObjectUtils.asMap(
        "public_id",
        "department" + "/" + multipartFile.getName()
    ));

    String departmentUrl = (String) response.get("secure_url");

    departmentDTO.setBackgroundImage(departmentUrl);
    DepartmentEntity departmentEntity = mapper.map(departmentDTO, DepartmentEntity.class);
    departmentRepository.save(departmentEntity);

    return mapper.map(departmentEntity, DepartmentDTO.class);
  }

  @Override
  public DepartmentDTO editDepartment(DepartmentDTO departmentDTO, MultipartFile multipartFile)
      throws IOException {
    DepartmentEntity departmentEntity = departmentRepository.getReferenceById(
        departmentDTO.getId());

    if(multipartFile != null ) {
      Map response = cloudinary.uploader().upload(multipartFile.getBytes(), ObjectUtils.asMap(
          "public_id",
          "department" + "/" + departmentDTO.getId() + "/" + multipartFile.getName()
      ));

      String departmentUrl = (String) response.get("secure_url");
      departmentEntity.setBackgroundImage(departmentUrl);
    }

    departmentEntity.setName(departmentDTO.getName());
    departmentRepository.save(departmentEntity);

    return mapper.map(departmentEntity, DepartmentDTO.class);
  }

  @Override
  public List<DepartmentDTO> deleteDepartment(Long departmentId) {
    List<MedicalExaminationEntity> medicalExaminationEntityList = medicalExaminationRepository.getAllByDepartmentId(
        departmentId);

    if (medicalExaminationEntityList.size() != 0) {
      throw new DeleteDepartmentException("You can't delete");
    }

    departmentRepository.delete(departmentRepository.getReferenceById(departmentId));
    return createListDepartmentDTO(departmentRepository.findAll());
  }
}
