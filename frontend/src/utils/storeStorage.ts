export const deleteValueStorage = (key: string) => {
  localStorage.removeItem(key)
}

export const getValueStorage = (key: string) => {
  let currentValue = JSON.parse(localStorage.getItem(key) ?? '{}')
  return currentValue
}

export const storeValueStorage = (key: string, value: any) => {
  localStorage.setItem(key, JSON.stringify(value))
}
