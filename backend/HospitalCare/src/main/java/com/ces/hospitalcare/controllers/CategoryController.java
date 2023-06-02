package com.ces.hospitalcare.controllers;
import com.ces.hospitalcare.dto.CategoryDTO;
import com.ces.hospitalcare.service.ICategoryService;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping(path = "/api/v1/guest/categories")
public class CategoryController {
  @Autowired
  private ICategoryService categoryService;

  @GetMapping(path = "")
  public List<CategoryDTO> getAllCategory() {
    return categoryService.getAllCategory();
  }
}
