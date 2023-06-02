package com.ces.hospitalcare.service;
import com.ces.hospitalcare.dto.DepartmentDTO;
import java.io.IOException;
import java.util.List;
import org.springframework.web.multipart.MultipartFile;

public interface IDepartmentService {
  List<DepartmentDTO> getAllDepartment();

  DepartmentDTO addDepartment(DepartmentDTO departmentDTO, MultipartFile multipartFile)
      throws IOException;

  DepartmentDTO editDepartment(DepartmentDTO departmentDTO, MultipartFile multipartFile)
      throws IOException;

  List<DepartmentDTO> deleteDepartment(Long departmentId);
}
