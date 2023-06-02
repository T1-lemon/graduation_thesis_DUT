import CloseIcon from '@mui/icons-material/Close'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import React, { useEffect, useState } from 'react'
import { toast } from 'react-toastify'

import ModalConfirm from '../../../components/ModalConfirm'
import { ITimeSlotResponse } from '../../../interface/TimeSlotInterfaces'
import { ITimeSlotRequest } from '../../../interface/TimeSlotInterfaces'
import { useAppDispatch } from '../../../redux/hooks'
import { deleteTimeSlotThunk } from '../../../redux/slices/timeSlotSlice'
import { editTimeSlotThunk } from '../../../redux/slices/timeSlotSlice'
import {
  MODAL_ACTION_CLOSE,
  MODAL_ACTION_CONFIRM,
  SUBMIT_MODAL_EDIT
} from '../../../utils/contant'
import {
  addHoursToDate,
  checkDayOfWeek,
  checkExistArrayDate,
  getAllHour,
  getTimeZone,
  subDate
} from '../../../utils/date'
import ModalEditTime from './ModalEditTime'

type Props = {
  timeSlotResponse: ITimeSlotResponse
  timeSlotsOfDay: ITimeSlotResponse[]
  dayOfWeek: Date
}

type SelectTimeEdit = {
  startTime: string
  endTime: string
  [key: string]: any
}

const arrStartTimeDefaut = getAllHour()

export default function TimeSlotItem(props: Props) {
  const dispatch = useAppDispatch()
  const { timeSlotsOfDay, timeSlotResponse, dayOfWeek } = props
  const { timeSlotDTO, appointmentId } = timeSlotResponse
  const [openModalConfirm, setOpenModalConfirm] = useState(false)
  const [openModalEdit, setOpenModalEdit] = useState(false)
  const [arrEndTime, setArrEndTime] = useState<Date[]>([])
  const [arrStartTimeCorrect, setArrStartTimeCorrect] = useState<Date[]>([])
  const [arrEndTimeCorrect, setArrEndTimeCorrect] = useState<Date[]>([])

  const arrStartTimeDefautDayOfWeek = arrStartTimeDefaut.map(
    (startTimeDate) => {
      startTimeDate.setDate(dayOfWeek.getDate())
      startTimeDate.setMonth(dayOfWeek.getMonth())
      startTimeDate.setFullYear(dayOfWeek.getFullYear())

      return startTimeDate
    }
  )

  const [selectTime, setSelectTime] = useState<SelectTimeEdit>({
    startTime: timeSlotDTO.startTime,
    endTime: `${addHoursToDate(
      new Date(timeSlotDTO.startTime),
      timeSlotDTO.duration
    )}`
  })

  useEffect(() => {
    setSelectTime({
      startTime: timeSlotDTO.startTime,
      endTime: `${addHoursToDate(
        new Date(timeSlotDTO.startTime),
        timeSlotDTO.duration
      )}`
    })

    createArrStartAndEndTime()
  }, [timeSlotsOfDay, dayOfWeek, timeSlotResponse])

  const toggleModalConfirm = () => {
    setOpenModalConfirm(!openModalConfirm)
  }

  const onModalDeleteTimeSlot = async (type: string) => {
    if (type === MODAL_ACTION_CONFIRM) {
      await dispatch(deleteTimeSlotThunk(timeSlotDTO.id))
      toast.success('Time Slot deleted successfully')
    }

    toggleModalConfirm()
  }

  const toggleModalEdit = () => {
    if (checkDayOfWeek(dayOfWeek)) setOpenModalEdit(!openModalEdit)
  }

  const handleChangeSelect = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const value = { ...selectTime }
    value[event.target.name] = event.target.value
    setSelectTime(value)
  }

  const handleEditTimeSlot = async (type: string) => {
    if (type === SUBMIT_MODAL_EDIT) {
      const minuteDuration: number = subDate(
        selectTime.startTime,
        selectTime.endTime
      )

      const startTimeUpdate = new Date(selectTime.startTime)

      const dataUpdate: ITimeSlotRequest = {
        startTime: startTimeUpdate.toISOString(),
        duration: minuteDuration
      }

      await dispatch(editTimeSlotThunk(timeSlotDTO.id, dataUpdate))
      toast.success('Time Slot Edit Successfully')
    }

    if (type === MODAL_ACTION_CLOSE) {
      setSelectTime({
        startTime: timeSlotDTO.startTime,
        endTime: `${addHoursToDate(
          new Date(timeSlotDTO.startTime),
          timeSlotDTO.duration
        )}`
      })
    }

    toggleModalEdit()
  }

  const createArrStartAndEndTime = () => {
    const arrStartTimeCheck: Date[] = []
    const arrEndTimeCheck: Date[] = []

    timeSlotsOfDay.forEach((item) => {
      const { timeSlotDTO } = item
      const dayByStartTime = new Date(timeSlotDTO.startTime)

      arrStartTimeCheck.push(dayByStartTime)
      arrEndTimeCheck.push(addHoursToDate(dayByStartTime, timeSlotDTO.duration))

      if (timeSlotDTO.duration === 60) {
        arrStartTimeCheck.push(addHoursToDate(dayByStartTime, 30))
        arrEndTimeCheck.push(addHoursToDate(dayByStartTime, 30))
      }
    })

    createArrStartTimeCorrect(arrStartTimeCheck, arrStartTimeDefautDayOfWeek)

    setArrEndTime(arrEndTimeCheck)
  }

  const createArrStartTimeCorrect = (
    arrStartTimeCheck: Date[],
    arrStartTimeMain: Date[]
  ) => {
    const arrTime: Date[] = arrStartTimeMain.filter((dayMainItem) => {
      return !checkExistArrayDate(arrStartTimeCheck, dayMainItem)
    })

    setArrStartTimeCorrect(arrTime)
  }

  const updateTime = () => {
    const startTimeDate = new Date(timeSlotDTO.startTime)
    const endTimeDate = addHoursToDate(startTimeDate, timeSlotDTO.duration)

    const arrEndTimeDelete: Date[] = []

    if (!checkExistArrayDate(arrStartTimeCorrect, startTimeDate))
      arrStartTimeCorrect.push(startTimeDate)

    arrEndTimeDelete.push(endTimeDate)
    if (timeSlotDTO.duration === 60) {
      const timeAdd: Date = addHoursToDate(startTimeDate, 30)
      if (!checkExistArrayDate(arrStartTimeCorrect, timeAdd)) {
        arrStartTimeCorrect.push(timeAdd)
      }

      arrEndTimeDelete.push(timeAdd)
    }

    const arrEndTimeUpdate: Date[] = arrEndTime.filter((endTimeItem) => {
      return !checkExistArrayDate(arrEndTimeDelete, endTimeItem)
    })

    setArrEndTime([...arrEndTimeUpdate])

    createEndTimeCorrectOriginal(arrEndTimeUpdate)
  }

  const createEndTimeCorrectOriginal = (arrEndTimeCheck: Date[]) => {
    const startTimeDate = new Date(selectTime.startTime)

    const arrEndTimeTemporary: Date[] = [
      addHoursToDate(startTimeDate, 30),
      addHoursToDate(startTimeDate, 60)
    ]

    const arrEndTimeCorrect: Date[] = arrEndTimeTemporary.filter(
      (endTimeTemporary) => {
        return !checkExistArrayDate(arrEndTimeCheck, endTimeTemporary)
      }
    )

    setArrEndTimeCorrect([...arrEndTimeCorrect])
  }

  const renderTimeSlot = () => {
    return appointmentId ? (
      <span className='cursor__noDrop'>{`${getTimeZone(
        timeSlotDTO.startTime
      )} - ${getTimeZone(
        addHoursToDate(new Date(timeSlotDTO.startTime), timeSlotDTO.duration)
      )}`}</span>
    ) : (
      <>
        <span
          onClick={() => {
            toggleModalEdit()
            updateTime()
          }}
        >{`${getTimeZone(timeSlotDTO.startTime)} - ${getTimeZone(
          addHoursToDate(new Date(timeSlotDTO.startTime), timeSlotDTO.duration)
        )}`}</span>
        {checkDayOfWeek(dayOfWeek) ? (
          <CloseIcon
            className='delete__time-icon'
            onClick={toggleModalConfirm}
          />
        ) : (
          ''
        )}
      </>
    )
  }
  return (
    <>
      <div className={`time__slot-item ${appointmentId ? 'appointment' : ''}`}>
        {renderTimeSlot()}
      </div>

      <ModalConfirm
        openModalConfirm={openModalConfirm}
        onAction={onModalDeleteTimeSlot}
        textBtn='Delete'
        backgroundColorBtnConfirm='#cb4c48'
        title='Delete Time Slot'
        contentBody={
          <>
            <h3>You are about to delete a time slot</h3>
            <p style={{ color: '#da4040' }}>
              This will delete your time slot. Are you sure ?
            </p>
          </>
        }
      />
      <ModalEditTime
        openModalEdit={openModalEdit}
        onAction={handleEditTimeSlot}
        handleChange={handleChangeSelect}
        valueStartTime={selectTime.startTime}
        valueEndTime={selectTime.endTime}
        arrStartTimeCorrect={arrStartTimeCorrect}
        arrEndTime={arrEndTime}
        arrEndTimeCorrectOriginal={arrEndTimeCorrect}
      />
    </>
  )
}
