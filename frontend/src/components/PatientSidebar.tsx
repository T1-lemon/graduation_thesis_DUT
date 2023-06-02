import CloseIcon from '@mui/icons-material/Close'
import { CircularProgress, Modal } from '@mui/material'
import { motion } from 'framer-motion'
import { useState } from 'react'
import { NavLink } from 'react-router-dom'

import '../assets/css/components/patient_sidebar.css'
import { DataUserProfilePicture } from '../interface/UsersInterface'
import { useAppDispatch, useAppSelector } from '../redux/hooks'
import { updateUserProfilePicture } from '../redux/thunk/userThunk'
import AvartarCustomize from './header/AvartarCustomize'

interface ProfileItemInterface {
  profileItem: string
  link: string
}
const profiles: ProfileItemInterface[] = [
  { profileItem: 'Profile Settings', link: '/user/profile-settings' },
  { profileItem: 'Appointments', link: '/user/appointments' }
]

function PatientSidebar() {
  const { currentUser } = useAppSelector((state) => state.auths)
  const dispatch = useAppDispatch()

  const [open, setOpen] = useState(false)
  const handleOpen = () => setOpen(true)
  const handleClose = () => {
    setOpen(false)
    setErrorMsg('')
  }

  const [errorMsg, setErrorMsg] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | string>('')
  const [loading, setLoading] = useState<boolean>(false)

  const handleFileChange = (event: React.FormEvent<HTMLInputElement>) => {
    const MAX_FILE_SIZE = 10240
    const files = event.currentTarget.files
    if (files && files.length > 0) {
      const file = files[0]
      const fileSizeKiloBytes = file.size / 1024

      if (fileSizeKiloBytes > MAX_FILE_SIZE) {
        setErrorMsg(
          'Your image is lager than maximum limit. Please choose a smaller image'
        )
        return
      }

      setErrorMsg('')
      setSelectedFile(file)
    }
  }

  const handleUpdateUserProfilePicture = () => {
    if (selectedFile) {
      const data: DataUserProfilePicture = {
        userId: currentUser.id,
        profilePicture: selectedFile
      }
      const fetchApiUpdateUserProfilePicture = async (
        data: DataUserProfilePicture
      ) => {
        setLoading(true)
        await dispatch(updateUserProfilePicture(data))
        setLoading(false)
        setSelectedFile('')
        setOpen(false)
      }
      fetchApiUpdateUserProfilePicture(data)
    }
  }

  return (
    <div className='patient__sidebar--container'>
      <div className='patient__div--header'>
        <div className='avatar__div--containers'>
          <AvartarCustomize
            profilePicture={currentUser.profilePicture}
            onClickBtn={handleOpen}
          />
        </div>
        <div className='patient__header--content'>
          <p className='patient__text--username'>{`${currentUser.firstName} ${currentUser.lastName}`}</p>
        </div>
      </div>
      <div className='profile__list'>
        <ul>
          {profiles.map((profile, index) => {
            return (
              <li className='profile__item' key={index}>
                <NavLink
                  to={profile.link}
                  className={({ isActive }) =>
                    isActive
                      ? 'profile__item--active'
                      : 'profile__item--default'
                  }
                >
                  {profile.profileItem}
                </NavLink>
              </li>
            )
          })}
        </ul>
      </div>
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby='modal-modal-title'
        aria-describedby='modal-modal-description'
      >
        <div className='modal__container'>
          <div className='modal__box--header'>
            <p className='modal__title'>Change your profile picture</p>
            <motion.button
              className='modal__button--close'
              whileTap={{ scale: 1.2 }}
              onClick={handleClose}
            >
              <CloseIcon fontSize='small' />
            </motion.button>
          </div>
          {loading ? (
            <div className='modal__loading'>
              <CircularProgress color='inherit' />
            </div>
          ) : (
            <>
              <p className='modal__title--input'>Choose File</p>
              <input
                type='file'
                id='avatar'
                name='avatar'
                accept='image/png, image/jpeg, image/jpg'
                max-size='1000'
                className='modal__input--avatar'
                onChange={handleFileChange}
              />
              <p className='modal__text--inform'>
                JPG, GIF or PNG. Max size of 10MB
              </p>
              {errorMsg ? (
                <p className='modal__text--error'>{errorMsg}</p>
              ) : (
                <></>
              )}
              <motion.button
                className='modal__button--upload'
                whileTap={{ scale: 1.1 }}
                onClick={handleUpdateUserProfilePicture}
              >
                <p>Upload Now</p>
              </motion.button>
            </>
          )}
        </div>
      </Modal>
    </div>
  )
}

export default PatientSidebar
