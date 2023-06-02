export const converGenderToBoolen = (gender: string): boolean => {
  return gender === 'female' ? true : false
}

export const convertGenderToString = (gender?: boolean): string => {
  return gender ? 'female' : 'male'
}
