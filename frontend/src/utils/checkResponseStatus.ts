export const checkResponseFailed = (status: number) => {
  return status >= 400 && status <= 600
}

export const checkResponseSuccess = (status: number) => {
  return status < 300 && status >= 200
}
