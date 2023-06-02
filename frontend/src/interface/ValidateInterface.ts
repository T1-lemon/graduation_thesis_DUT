export interface FormLoginValues {
  email: string
  password: string
}

export interface FormRegisterValues {
  email: string
  password: string
  firstName: string
  lastName: string
  confirmPassword?: string
  role: string
}

export interface RegisterSuccessResponse {
  message: string
  status: number
}
