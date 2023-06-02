package com.ces.hospitalcare.controllers;
import com.ces.hospitalcare.dto.DepartmentDTO;
import com.ces.hospitalcare.service.IDepartmentService;
import java.io.IOException;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/v1/department")
public class DepartmentController {
  @Autowired
  private IDepartmentService departmentService;

  @GetMapping(value = "")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
  public List<DepartmentDTO> getAllDepartment() {
    return departmentService.getAllDepartment();
  }

  @RequestMapping(value = "", method = RequestMethod.POST, consumes = {"multipart/form-data"})
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
  public DepartmentDTO addDepartment(@RequestPart("data") DepartmentDTO departmentDTO, @RequestPart("file") MultipartFile multipartFile) throws IOException {
    System.out.println("abc");
    return departmentService.addDepartment(departmentDTO, multipartFile);
  }

  @RequestMapping(value = "/{id}", method = RequestMethod.POST, consumes = {"multipart/form-data"})
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
  public DepartmentDTO editDepartment(@RequestPart("data") DepartmentDTO departmentDTO, @RequestPart(value = "file", required = false) MultipartFile multipartFile,
      @PathVariable("id") Long departmentId) throws IOException {
    departmentDTO.setId(departmentId);
    return departmentService.editDepartment(departmentDTO, multipartFile);
  }


  @DeleteMapping(value = "/{id}")
  @PreAuthorize("hasAnyAuthority('ROLE_ADMIN')")
  public List<DepartmentDTO> deleteDepartment(@PathVariable("id") Long departmentId) {
    return departmentService.deleteDepartment(departmentId);
  }
}
