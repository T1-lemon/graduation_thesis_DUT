export interface AdminDepartmentInterface {
  id: number
  backgroundImage: string
  name: string
}

export interface AdminDoctorInterface {
  doctorInfor: AdminUserInterface
  medicalExamination: AdminMedicalExaminationInterface
  department: AdminDepartmentInterface
}

export interface AdminMedicalExaminationInterface {
  id: number
  title: string
  description: string
  shortDescription: string
  examinationPrice: number
  image: string
}

export interface AdminUserInterface {
  id: number
  firstName: string
  lastName: string
  profilePicture: string
  age: number
  email: string
  address: string
  phoneNumber: string
  gender: boolean
  password: string
  confirmPassword: string
  departmentId: number
}

export interface DataAdminSetDoctorProfile {
  doctorId: number
  firstName: string
  lastName: string
  gender: boolean
  address: string
  phoneNumber: string
  age: number
  departmentId: string
  email: string
}

export interface FormAdminSetDoctorProfileValues {
  firstName: string
  lastName: string
  gender: string
  email: string
  address: string
  phoneNumber: string
  age: string
  departmentId: string
  password: string
  confirmPassword: string
  role: string
}
