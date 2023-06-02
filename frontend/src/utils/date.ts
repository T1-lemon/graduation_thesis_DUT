import { ITimeSlot } from '../interface/TimeSlotInterfaces';


export const addHoursToDate = (objDate: Date, minuteDuration: number): Date => {
  var numberOfMilliseconds = objDate.getTime()
  var addMilliseconds = minuteDuration * 60 * 1000
  var newDateObj = new Date(numberOfMilliseconds + addMilliseconds)
  return newDateObj
}

export const subHoursToDate = (objDate: Date, minuteDuration: number): Date => {
  let numberOfMlSeconds = objDate.getTime()
  let addMlSeconds = minuteDuration * 60 * 1000
  let newDateObj = new Date(numberOfMlSeconds - addMlSeconds)
  return newDateObj
}

export const checkExistArrayDate = (arrDateCheck: Date[], date: Date) => {
  const checkExist = arrDateCheck.some(
    (timeItem) => timeItem.getTime() === date.getTime()
  )
  return checkExist
}

export const compareDate = (
  objDateA: Date | string,
  objDateB: Date | string
): boolean => {
  const dateA = new Date(objDateA)
  const dateB = new Date(objDateB)
  dateA.setHours(0, 0, 0, 0)
  dateB.setHours(0, 0, 0, 0)

  return dateA.getTime() === dateB.getTime()
}

export const subDate = (
  startDate: Date | string,
  endDate: Date | string
): number => {
  const objStartDate = new Date(startDate)
  const objEndDate = new Date(endDate)
  objStartDate.setDate(1)
  objStartDate.setMonth(4)
  objStartDate.setFullYear(2023)

  objEndDate.setDate(1)
  objEndDate.setMonth(4)
  objEndDate.setFullYear(2023)

  const miliSecondsDuration = objEndDate.getTime() - objStartDate.getTime()
  const minuteDuration = miliSecondsDuration / (60 * 1000)
  return minuteDuration
}

export const formatDate = (date: Date): string => {
  const dayOfWeek = new Intl.DateTimeFormat('en-US', {
    weekday: 'long'
  }).format(date)
  const dateFormat = date.toLocaleDateString('en-US')
  const rs = `${dayOfWeek} - ${dateFormat}`
  return rs
}

export const getAllDateOfCurrentWeek = (): string[] => {
  let currentDay = new Date()
  let week = []

  for (let i = 1; i <= 6; i++) {
    let first = currentDay.getDate() - currentDay.getDay() + i
    let day = new Date(currentDay.setDate(first)).toISOString()
    week.push(day)
  }
  return week
}

export const getAllHour = (): Date[] => {
  let date7Hour = new Date('2023-05-01T00:00:00.000+00:00')
  let day1 = new Date('2023-05-01T05:00:00.000+00:00') //12h
  let day2 = new Date('2023-05-01T06:30:00.000+00:00') //13h30

  let arrHour: Date[] = [date7Hour]

  for (let i = 1; i <= 19; i++) {
    let day = new Date(addHoursToDate(arrHour[i - 1], 30))
    arrHour.push(day)
  }

  const arrStartTime: Date[] = arrHour.filter((item) => {
    return item < day1 || item > day2
  })

  return arrStartTime
}

export const getDateOfWeek = (date: string): string => {
  const dayOfWeek = new Intl.DateTimeFormat('en-US', {
    weekday: 'long'
  }).format(new Date(date))

  return dayOfWeek
}

export const getTimeSlotsAfterCurrentDay = (
  arrTimeSlots: ITimeSlot[]
): ITimeSlot[] => {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  return arrTimeSlots.filter((item) => {
    let date = new Date(item.startTime)
    return date > today
  })
}

export const getTimeSlotsDateFirst = (
  arrTimeSlots: ITimeSlot[],
  dateFirst: Date
): ITimeSlot[] => {
  dateFirst?.setHours(0, 0, 0, 0)
  return arrTimeSlots.filter((item) => {
    let date = new Date(item.startTime)
    date.setHours(0, 0, 0, 0)
    return dateFirst.getTime() === date.getTime()
  })
}

export const getTimeZone = (dateString: string | Date): string => {
  const date = new Date(dateString)
  return new Intl.DateTimeFormat('vi-VN', {
    hour: 'numeric',
    minute: 'numeric'
  }).format(date)
}

export const removeDuplicatesDate = (array: Date[]): Date[] => {
  return array
    .map(function (date) {
      return date.getTime()
    })
    .filter(function (date, i, array) {
      return array.indexOf(date) === i
    })
    .map(function (time) {
      return new Date(time)
    })
}

export const sortDate = (array: Date[]): Date[] => {
  return array.sort(function (a: Date, b: Date) {
    return a.getTime() - b.getTime()
  })
}

export const sortStartTime = (arrTimeSlots: ITimeSlot[]): ITimeSlot[] => {
  return arrTimeSlots.sort(function (a: ITimeSlot, b: ITimeSlot) {
    return new Date(a.startTime).getTime() - new Date(b.startTime).getTime()
  })
}

export const checkDayOfWeek = (dayOfWeek: string|Date): boolean => {
  const today = new Date()
  today.setHours(0, 0, 0, 0)

  const objDayOfWeek = new Date(dayOfWeek)
  objDayOfWeek.setHours(0, 0, 0, 0)

  return today <= objDayOfWeek
}
