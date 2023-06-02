import CloseIcon from '@mui/icons-material/Close'
import { Box, Modal } from '@mui/material'

import '../../assets/css/templates/admin_template.css'
import ButtonCustomize from '../../components/ButtonCustomize'

type Props = {
  classNameHeader: string
  classNameContainer?: string
  open: boolean
  title: string
  titleDelete?: string
  desDelete?: string
  handleClose: () => void
  bodyModal?: JSX.Element
  footerModal?: JSX.Element
}

function AdminModelCustomize(props: Props) {
  const {
    classNameHeader,
    classNameContainer,
    open,
    title,
    handleClose,
    bodyModal,
    footerModal,
    titleDelete,
    desDelete
  } = props

  return (
    <div>
      <Modal open={open} onClose={handleClose}>
        <Box
          className={`admin__modal--container ${
            classNameContainer ? classNameContainer : ''
          }`}
        >
          <Box className={`admin__modal--header ${classNameHeader}`}>
            <p className='admin__model--title'>{title}</p>
            <ButtonCustomize
              icon={<CloseIcon fontSize='small' />}
              className='btn__alert--close'
              onClickBtn={handleClose}
            />
          </Box>
          <Box className='admin__modal--body'>
            {bodyModal ? (
              <div>{bodyModal}</div>
            ) : (
              <div className='admin__modal--delete'>
                <p className='admin__title--delete'>{titleDelete}</p>
                <p className='admin__des--delete'>{desDelete}</p>
              </div>
            )}
          </Box>
          {footerModal ? (
            <Box className='admin__model--footer'>
              <div>{footerModal}</div>
            </Box>
          ) : (
            <></>
          )}
        </Box>
      </Modal>
    </div>
  )
}

export default AdminModelCustomize
